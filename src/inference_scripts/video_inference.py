import argparse
import functools
import logging
import os
import os.path

import tensorflow as tf

import cameralib
import options
import poseviz
import video_io
from options import FLAGS


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--video-dir', type=str)
    parser.add_argument('--video-filenames', type=str)
    parser.add_argument('--viz-downscale', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-detections', type=int, default=-1)
    parser.add_argument('--out-video-path', type=str)
    parser.add_argument('--write-video', action=options.BoolAction)
    parser.add_argument('--num-aug', type=int, default=5)
    parser.add_argument('--fov', type=float, default=55)
    parser.add_argument('--skeleton', type=str, default='smpl+head_30')
    parser.add_argument('--model-name', type=str, default='')
    options.initialize(parser)
    logging.getLogger('absl').setLevel('ERROR')
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    initialize()
    model = tf.saved_model.load(FLAGS.model_path)
    joint_names = model.per_skeleton_joint_names[FLAGS.skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[FLAGS.skeleton].numpy()
    predict_fn = functools.partial(
        model.detect_poses_batched, default_fov_degrees=FLAGS.fov,
        detector_threshold=0.5, detector_flip_aug=True,
        num_aug=FLAGS.num_aug, detector_nms_iou_threshold=0.8, internal_batch_size=64 * 3,
        skeleton=FLAGS.skeleton, suppress_implausible_poses=True, antialias_factor=2,
        max_detections=FLAGS.max_detections)

    video_filenames = FLAGS.video_filenames.split(',')
    video_paths = [f'{FLAGS.video_dir}/{f}' for f in video_filenames]

    with poseviz.PoseViz(
            joint_names, joint_edges,
            downscale=FLAGS.viz_downscale, write_video=FLAGS.write_video) as viz:
        for video_path in video_paths:
            predict_video(predict_fn, video_path, viz)


def predict_video(predict_fn, video_path, viz):
    frame_batch_ds, frame_batch_cpu = video_io.video_as_tf_dataset(
        video_path, batch_size=FLAGS.batch_size, prefetch_gpu=1, tee_cpu=True)

    viz.new_sequence()
    if FLAGS.write_video:
        filename = os.path.basename(video_path)
        out_video_path = os.path.join(FLAGS.out_video_path, filename)
        viz.start_new_video(out_video_path, fps=24)

    camera = cameralib.Camera.from_fov(FLAGS.fov, frame_batch_ds.element_spec.shape[1:3])
    for frame_b, frame_b_cpu in zip(frame_batch_ds, frame_batch_cpu):
        pred = predict_fn(frame_b)
        for frame, boxes, poses3d in zip(
                frame_b_cpu, pred['boxes'].numpy(), pred['poses3d'].numpy()):
            viz.update(frame, boxes, poses3d, camera)


if __name__ == '__main__':
    main()
