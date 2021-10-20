#!/usr/bin/env python3
import video_io
import argparse
import functools

import h5py
import numpy as np
import tensorflow as tf

import data.datasets3d
import data.h36m
import data.mpi_inf_3dhp
import options
import paths
import tfu
import util
from options import FLAGS
import poseviz


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--out-video-dir', type=str)
    parser.add_argument('--num-aug', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--viz', action=options.BoolAction)
    options.initialize(parser)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    initialize()
    model = tf.saved_model.load(FLAGS.model_path)
    ji3d = get_joint_info(model, skeleton='mpi_inf_3dhp_17')

    predict_fn = functools.partial(
        model.detect_poses_batched, internal_batch_size=0, num_aug=FLAGS.num_aug,
        detector_threshold=0, detector_flip_aug=True, antialias_factor=2, max_detections=1,
        suppress_implausible_poses=False, skeleton='mpi_inf_3dhp_17')

    viz = poseviz.PoseViz(
        ji3d.names, ji3d.stick_figure_edges, write_video=bool(FLAGS.out_video_dir),
        world_up=(0, 1, 0), downscale=4, queue_size=2 * FLAGS.batch_size) if FLAGS.viz else None

    image_relpaths_all = []
    coords_all = []
    cam1_4 = data.mpi_inf_3dhp.get_test_camera_subj1_4()
    cam5_6 = data.mpi_inf_3dhp.get_test_camera_subj5_6()

    for subj in range(1, 7):
        if FLAGS.viz:
            viz.new_sequence()
            if FLAGS.out_video_dir:
                viz.start_new_video(f'{FLAGS.out_video_dir}/TS{subj}.mp4', fps=50)

        camera = cam1_4 if subj <= 4 else cam5_6
        with h5py.File(f'{paths.DATA_ROOT}/3dhp/TS{subj}/annot_data.mat', 'r') as m:
            valid_frames = np.where(m['valid_frame'][:, 0])[0]

        frame_relpaths = [f'3dhp/TS{subj}/imageSequence/img_{i + 1:06d}.jpg' for i in valid_frames]
        frame_paths = [f'{paths.DATA_ROOT}/{p}' for p in frame_relpaths]
        frames_gpu, frames_cpu = video_io.image_files_as_tf_dataset(
            frame_paths, batch_size=FLAGS.batch_size, prefetch_gpu=2, tee_cpu=FLAGS.viz)

        coords3d_pred_world = predict_sequence(
            predict_fn, frames_gpu, frames_cpu, len(frame_paths), camera, viz)
        image_relpaths_all.append(frame_relpaths)
        coords_all.append(coords3d_pred_world)

    np.savez(
        FLAGS.output_path, image_path=np.concatenate(image_relpaths_all, axis=0),
        coords3d_pred_world=np.concatenate(coords_all, axis=0))

    if FLAGS.viz:
        viz.close()


def predict_sequence(predict_fn, frames_gpu, frames_cpu, n_frames, camera, viz):
    predict_fn = functools.partial(
        predict_fn,
        intrinsic_matrix=camera.intrinsic_matrix[np.newaxis],
        distortion_coeffs=camera.get_distortion_coeffs()[np.newaxis],
        extrinsic_matrix=camera.get_extrinsic_matrix()[np.newaxis],
        world_up_vector=camera.world_up)
    progbar = util.progressbar(total=n_frames)
    pose_batches = []

    for frames_b, frames_b_cpu in zip(frames_gpu, frames_cpu):
        pred = predict_fn(frames_b)
        pred = tf.nest.map_structure(lambda x: tf.squeeze(x, 1).numpy(), pred)
        pose_batches.append(pred['poses3d'])
        progbar.update(frames_b.shape[0])

        if FLAGS.viz:
            for frame, box, pose3d in zip(frames_b_cpu, pred['boxes'], pred['poses3d']):
                viz.update(frame, box[np.newaxis], pose3d[np.newaxis], camera)

    return np.concatenate(pose_batches, axis=0)


def get_joint_info(model, skeleton):
    joint_names = [b.decode('utf8') for b in model.per_skeleton_joint_names[skeleton].numpy()]
    edges = model.per_skeleton_joint_edges[skeleton].numpy()
    return data.datasets3d.JointInfo(joint_names, edges)


if __name__ == '__main__':
    main()
