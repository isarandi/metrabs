#!/usr/bin/env python3
import os

import cv2
import tensorflow as tf

import poseviz


def main():
    model = tf.saved_model.load(download_model('metrabs_rn18_y4'))
    skeleton = 'smpl+head_30'
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()
    viz = poseviz.PoseViz(joint_names, joint_edges)

    for frame in frames_from_webcam():
        pred = model.detect_poses(
            frame, skeleton=skeleton, default_fov_degrees=55, detector_threshold=0.5)
        camera = poseviz.Camera.from_fov(55, frame.shape[:2])
        viz.update(frame, pred['boxes'], pred['poses3d'], camera)


def frames_from_webcam():
    cap = cv2.VideoCapture(0)
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
