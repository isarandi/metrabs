import argparse

import numpy as np
import tensorflow as tf

import options
from options import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-model-path', type=str, required=True)
    parser.add_argument('--detector-path', type=str, required=True)
    parser.add_argument('--output-model-path', type=str, required=True)
    options.initialize(parser)

    detector = tf.saved_model.load(FLAGS.detector_path)
    pose_estimator = tf.saved_model.load(FLAGS.input_model_path)
    combined_model = DetectorAndPoseEstimator(detector, pose_estimator)
    tf.saved_model.save(combined_model, FLAGS.output_model_path)


class DetectorAndPoseEstimator(tf.Module):
    def __init__(self, detector, pose_estimator):
        super().__init__()
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.joint_names = pose_estimator.joint_names
        self.joint_edges = pose_estimator.joint_edges

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32)])
    def predict_single_image(
            self, image, intrinsics=((),), detector_threshold=0.5,
            detector_nms_iou_threshold=0.4, detector_flip_aug=False, internal_batch_size=64,
            n_aug=5):
        detections, poses, poses2d = self.predict_multi_image(
            image[np.newaxis], intrinsics[np.newaxis], detector_threshold,
            detector_nms_iou_threshold, detector_flip_aug, internal_batch_size, n_aug)
        return detections[0], poses[0], poses2d[0]

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32)])
    def predict_multi_image(
            self, image, intrinsics=(((),),), detector_threshold=0.5,
            detector_nms_iou_threshold=0.4, detector_flip_aug=False, internal_batch_size=64,
            n_aug=5):
        detections = self.detector(
            image, detector_threshold, detector_nms_iou_threshold, detector_flip_aug)

        # Assume 50 degree field of view if intrinsics not given
        if tf.size(intrinsics) == 0:
            imshape = tf.cast(tf.shape(image)[1:3], tf.float32)
            focal_length = (
                    tf.reduce_max(imshape) /
                    tf.constant(np.tan(np.deg2rad(50) / 2) * 2, tf.float32))
            intrinsics = tf.convert_to_tensor(
                [[[focal_length, 0, imshape[1] / 2],
                  [0, focal_length, imshape[0] / 2],
                  [0, 0, 1]]], tf.float32)

        if tf.shape(intrinsics)[0] == 1:
            n_images = tf.shape(image)[0]
            intrinsics = tf.repeat(intrinsics, n_images, axis=0)

        poses = self.pose_estimator.predict_multi_image(
            image, intrinsics, detections[..., :4], internal_batch_size, n_aug)

        # Also return the 2D projected poses for convenience
        n_poses_per_image = poses.row_lengths()
        poses_flat = poses.flat_values
        intrinsics_repeat = tf.repeat(intrinsics, n_poses_per_image, axis=0)
        poses2d_flat = tf.einsum(
            '...nk,...jk->...nj', poses_flat / poses_flat[..., 2:], intrinsics_repeat)[..., :2]
        poses2d = tf.RaggedTensor.from_row_lengths(poses2d_flat, n_poses_per_image)
        return detections, poses, poses2d


if __name__ == '__main__':
    main()
