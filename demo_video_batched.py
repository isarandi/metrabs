#!/usr/bin/env python3
import os
import sys

import cv2
import tensorflow as tf

import poseviz
import cameralib


def main():
    model = tf.saved_model.load(download_model('metrabs_eff2l_y4_360'))
    skeleton = 'smpl+head_30'
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()
    viz = poseviz.PoseViz(joint_names, joint_edges)
    frame_batches = tf.data.Dataset.from_generator(
        frames_from_video, tf.uint8, [None, None, 3]).batch(32).prefetch(1)

    for frame_batch in frame_batches:
        pred = model.detect_poses_batched(frame_batch, skeleton=skeleton, default_fov_degrees=55)
        camera = cameralib.Camera.from_fov(55, frame_batch.shape[1:3])
        for frame, boxes, poses3d in zip(
                frame_batch.numpy(), pred['boxes'].numpy(), pred['poses3d'].numpy()):
            viz.update(frame, boxes, poses3d, camera)
    viz.close()


def frames_from_video():
    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    while (frame_bgr := cap.read()[1]) is not None:
        yield frame_bgr[..., ::-1]


def download_model(model_type):
    server_prefix = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'
    model_zippath = tf.keras.utils.get_file(
        origin=f'{server_prefix}/{model_type}.zip',
        extract=True, cache_subdir='models')
    model_path = os.path.join(os.path.dirname(model_zippath), model_type)
    return model_path


if __name__ == '__main__':
    main()
