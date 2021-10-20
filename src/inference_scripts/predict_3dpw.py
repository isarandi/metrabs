#!/usr/bin/env python3
import argparse
import functools
import glob
import os
import pickle

import numpy as np
import scipy.optimize
import tensorflow as tf

import cameralib
import data.datasets3d
import options
import paths
import poseviz
import util
import video_io
from options import FLAGS, logger


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--out-video-dir', type=str)
    parser.add_argument('--default-fov', type=float, default=55)
    parser.add_argument('--num-aug', type=int, default=5)
    parser.add_argument('--real-intrinsics', action=options.BoolAction)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--internal-batch-size', type=int, default=128)
    parser.add_argument('--antialias-factor', type=int, default=2)
    parser.add_argument('--viz', action=options.BoolAction)
    options.initialize(parser)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    initialize()
    model = tf.saved_model.load(FLAGS.model_path)
    ji3d = get_joint_info(model, 'smpl_24')

    predict_fn = functools.partial(
        model.detect_poses_batched, internal_batch_size=FLAGS.internal_batch_size,
        detector_threshold=0.2, detector_nms_iou_threshold=0.7, detector_flip_aug=True,
        antialias_factor=FLAGS.antialias_factor, num_aug=FLAGS.num_aug, suppress_implausible_poses=False,
        default_fov_degrees=FLAGS.default_fov, skeleton='smpl_24')

    seq_filepaths = sorted(glob.glob(f'{paths.DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl'))
    seq_names = [os.path.basename(p).split('.')[0] for p in seq_filepaths]

    ji2d = data.datasets3d.JointInfo(
        'nose,neck,rsho,relb,rwri,lsho,lelb,lwri,rhip,rkne,rank,lhip,lkne,lank,reye,leye,lear,rear')
    viz = poseviz.PoseViz(
        ji3d.names, ji3d.stick_figure_edges, write_video=bool(FLAGS.out_video_dir),
        queue_size=2 * FLAGS.batch_size) if FLAGS.viz else None

    for seq_name, seq_filepath in util.progressbar(zip(seq_names, seq_filepaths)):
        if FLAGS.viz:
            viz.new_sequence()
            if FLAGS.out_video_dir:
                viz.start_new_video(f'{FLAGS.out_video_dir}/{seq_name}.mp4', fps=25)

        already_done_files = glob.glob(f'{FLAGS.output_path}/*/*.pkl')
        if any(seq_name in p for p in already_done_files):
            logger.info(f'{seq_name} has been processed already.')
            continue
        logger.info(f'Predicting {seq_name}...')
        frame_paths = sorted(
            glob.glob(f'{paths.DATA_ROOT}/3dpw/imageFiles/{seq_name}/image_*.jpg'))
        poses2d_true = get_poses2d_3dpw(seq_name)
        frames_gpu, frames_cpu = video_io.image_files_as_tf_dataset(
            frame_paths, batch_size=FLAGS.batch_size, prefetch_gpu=2, tee_cpu=True)

        camera = get_3dpw_camera(seq_filepath) if FLAGS.real_intrinsics else None
        tracks = predict_sequence(
            predict_fn, frames_gpu, frames_cpu, len(frame_paths), poses2d_true, ji2d, ji3d, viz,
            camera)
        save_result_file(seq_name, FLAGS.output_path, tracks)

    if viz is not None:
        viz.close()


def predict_sequence(
        predict_fn, frame_batches_gpu, frame_batches_cpu, n_frames, poses2d_true, joint_info2d,
        joint_info3d, viz,
        camera=None):
    n_tracks = poses2d_true.shape[1]
    prev_poses2d_pred_ordered = np.zeros((n_tracks, joint_info3d.n_joints, 2))
    tracks = [[] for _ in range(n_tracks)]

    if camera is not None:
        predict_fn = functools.partial(predict_fn, intrinsic_matrix=camera.intrinsic_matrix)

    progbar = util.progressbar(total=n_frames)
    i_frame = 0
    for frames_gpu, frames_cpu in zip(frame_batches_gpu, frame_batches_cpu):
        pred = predict_fn(frames_gpu)
        pred = tf.nest.map_structure(lambda x: x.numpy(), pred)

        for frame, boxes, poses3d, poses2d in zip(
                frames_cpu, pred['boxes'], pred['poses3d'], pred['poses2d']):
            poses3d_ordered, prev_poses2d_pred_ordered = associate_predictions(
                poses3d, poses2d, poses2d_true[i_frame], prev_poses2d_pred_ordered,
                joint_info3d, joint_info2d)
            for pose, track in zip(poses3d_ordered, tracks):
                if not np.any(np.isnan(pose)):
                    track.append((i_frame, pose))

            poses3d = np.array([t[-1][1] for t in tracks if t])
            if viz is not None:
                if camera is None:
                    camera = cameralib.Camera.from_fov(FLAGS.default_fov, frame.shape)
                viz.update(frame, boxes[:, :4], poses3d, camera)
            progbar.update(1)
            i_frame += 1

    return tracks


def get_3dpw_camera(seq_filepath):
    with open(seq_filepath, 'rb') as f:
        intr = pickle.load(f, encoding='latin1')['cam_intrinsics']
        return cameralib.Camera(intrinsic_matrix=intr, world_up=[0, -1, 0])


def get_poses2d_3dpw(seq_name):
    seq_filepaths = glob.glob(f'{paths.DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl')
    filepath = next(p for p in seq_filepaths if os.path.basename(p) == f'{seq_name}.pkl')
    with open(filepath, 'rb') as f:
        seq = pickle.load(f, encoding='latin1')
    return np.transpose(np.array(seq['poses2d']), [1, 0, 3, 2])  # [Frame, Track, Joint, Coord]


def pose2d_auc(pose2d_pred, pose2d_true, prev_pose2d_pred, joint_info3d, joint_info2d):
    pose2d_true = pose2d_true.copy()
    pose2d_true[pose2d_true[:, 2] < 0.2] = np.nan
    selected_joints = 'lsho,rsho,lelb,relb,lhip,rhip,lkne,rkne'.split(',')
    indices_true = [joint_info2d.ids[name] for name in selected_joints]
    indices_pred = [joint_info3d.ids[name] for name in selected_joints]
    size = np.linalg.norm(pose2d_pred[joint_info3d.ids.rsho] - pose2d_pred[joint_info3d.ids.lhip])
    dist = np.linalg.norm(pose2d_true[indices_true, :2] - pose2d_pred[indices_pred], axis=-1)
    if np.count_nonzero(~np.isnan(dist)) < 5:
        dist = np.linalg.norm(prev_pose2d_pred[indices_pred] - pose2d_pred[indices_pred], axis=-1)
    return np.nanmean(np.maximum(0, 1 - dist / size))


def associate_predictions(
        poses3d_pred, poses2d_pred, poses2d_true, prev_poses2d_pred_ordered, joint_info3d,
        joint_info2d):
    auc_matrix = np.array([
        [pose2d_auc(pose_pred, pose_true, prev_pose, joint_info3d, joint_info2d)
         for pose_pred in poses2d_pred]
        for pose_true, prev_pose in zip(poses2d_true, prev_poses2d_pred_ordered)])

    true_indices, pred_indices = scipy.optimize.linear_sum_assignment(-auc_matrix)
    n_true_poses = len(poses2d_true)

    result = np.full((n_true_poses, joint_info3d.n_joints, 3), np.nan)
    poses2d_pred_ordered = np.array(prev_poses2d_pred_ordered).copy()
    for ti, pi in zip(true_indices, pred_indices):
        result[ti] = poses3d_pred[pi]
        poses2d_pred_ordered[ti] = poses2d_pred[pi]

    return result, poses2d_pred_ordered


def complete_track(track, n_frames):
    track_dict = dict(track)
    result = []
    for i in range(n_frames):
        if i in track_dict:
            result.append(track_dict[i])
        elif result:
            result.append(result[-1])
        else:
            result.append(np.full_like(track[0][1], fill_value=np.nan))
    return result


def save_result_file(seq_name, pred_dir, tracks):
    seq_filepaths = glob.glob(f'{paths.DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl')
    seq_path = next(p for p in seq_filepaths if os.path.basename(p) == f'{seq_name}.pkl')
    rel_path = '/'.join(util.split_path(seq_path)[-2:])
    out_path = f'{pred_dir}/{rel_path}'
    n_frames = len(glob.glob(f'{paths.DATA_ROOT}/3dpw/imageFiles/{seq_name}/image_*.jpg'))
    coords3d_raw = np.array([complete_track(track, n_frames) for track in tracks]) / 1000
    util.dump_pickle(dict(jointPositions=coords3d_raw), out_path)


def get_joint_info(model, skeleton):
    joint_names = [b.decode('utf8') for b in model.per_skeleton_joint_names[skeleton].numpy()]
    edges = model.per_skeleton_joint_edges[skeleton].numpy()
    return data.datasets3d.JointInfo(joint_names, edges)


if __name__ == '__main__':
    main()
