#!/usr/bin/env python3
import argparse
import logging

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import data.datasets3d
import options
from options import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-model-path', type=str, required=True)
    parser.add_argument('--output-model-path', type=str, required=True)
    parser.add_argument('--antialias-factor', type=int, default=1)
    options.initialize(parser)
    pose_estimator = Pose3dEstimator(
        FLAGS.input_model_path, antialias_factor=FLAGS.antialias_factor)
    tf.saved_model.save(pose_estimator, FLAGS.output_model_path)
    logging.info(f'Full image model has been exported to {FLAGS.output_model_path}')


class Pose3dEstimator(tf.Module):
    def __init__(self, model_path, antialias_factor=4):
        super().__init__()
        self.antialias_factor = antialias_factor
        self.crop_model = tf.saved_model.load(model_path)
        self.crop_side = 256
        self.joint_names = self.crop_model.joint_names
        self.joint_edges = self.crop_model.joint_edges
        joint_names = [b.decode('utf8') for b in self.joint_names.numpy()]
        self.joint_info = data.datasets3d.JointInfo(joint_names, self.joint_edges.numpy())

        self.__call__.get_concrete_function(
            tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(None, 3, 3), dtype=tf.float32),
            tf.RaggedTensorSpec(shape=(None, None, 4), ragged_rank=1, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32))

        self.__call__.get_concrete_function(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(3, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32))

    @tf.function
    def __call__(self, image, intrinsic_matrix, boxes, internal_batch_size=64, n_aug=5):
        if image.shape.rank == 3:
            return self.predict_single_image(
                image, intrinsic_matrix, boxes, internal_batch_size, n_aug)
        else:
            return self.predict_multi_image(
                image, intrinsic_matrix, boxes, internal_batch_size, n_aug)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(3, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32)])
    def predict_single_image(self, image, intrinsic_matrix, boxes, internal_batch_size=64, n_aug=5):
        if tf.size(boxes) == 0:
            return tf.zeros(shape=(0, self.joint_info.n_joints, 3))
        ragged_boxes = tf.RaggedTensor.from_tensor(boxes[np.newaxis])
        return self.predict_multi_image(
            image[np.newaxis], intrinsic_matrix[np.newaxis], ragged_boxes, internal_batch_size,
            n_aug)[0]

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(None, 3, 3), dtype=tf.float32),
        tf.RaggedTensorSpec(shape=(None, None, 4), ragged_rank=1, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32)])
    def predict_multi_image(self, image, intrinsic_matrix, boxes, internal_batch_size=64, n_aug=5):
        """Estimate 3D human poses in camera space for multiple bounding boxes specified
        for an image.
        """
        n_images = tf.shape(image)[0]
        if tf.size(boxes) == 0:
            # Special case for zero boxes provided
            result_flat = tf.zeros(shape=(0, self.joint_info.n_joints, 3))
            return tf.RaggedTensor.from_row_lengths(result_flat, tf.zeros(n_images, tf.int64))

        boxes_flat = boxes.flat_values
        n_box_per_image = boxes.row_lengths()
        image_id_per_box = boxes.value_rowids()
        n_total_boxes = tf.shape(boxes_flat)[0]
        # image = (tf.cast(image, tf.float32) / np.float32(255))# ** 2.2

        if tf.shape(intrinsic_matrix)[0] == 1:
            intrinsic_matrix = tf.repeat(intrinsic_matrix, n_images, axis=0)

        Ks = tf.repeat(intrinsic_matrix, n_box_per_image, axis=0)
        gammas = tf.cast(tf.linspace(0.6, 1.0, n_aug), tf.float16)
        angle_range = np.float32(np.deg2rad(25))
        angles = tf.linspace(-angle_range, angle_range, n_aug)
        scales = tf.concat([
            linspace_noend(0.8, 1.0, n_aug // 2),
            tf.linspace(1.0, 1.1, n_aug - n_aug // 2)], axis=0)
        flips = (tf.range(n_aug) - n_aug // 2) % 2 != 0
        flipmat = tf.constant([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
        maybe_flip = tf.where(flips[:, np.newaxis, np.newaxis], flipmat, tf.eye(3))
        rotmat = rotation_mat_zaxis(-angles)

        crops_per_batch = internal_batch_size // n_aug

        if crops_per_batch == 0:
            # No batching
            results_flat = self.predict_single_batch(
                image, Ks, boxes_flat, image_id_per_box, n_aug, rotmat, maybe_flip, flips, scales,
                gammas)
            return tf.RaggedTensor.from_row_lengths(results_flat, n_box_per_image)

        n_batches = tf.cast(tf.math.ceil(n_total_boxes / crops_per_batch), tf.int32)
        result_batches = tf.TensorArray(
            tf.float32, size=n_batches, element_shape=(None, self.joint_info.n_joints, 3),
            infer_shape=False)

        for i in tf.range(n_batches):
            box_batch = boxes_flat[i * crops_per_batch:(i + 1) * crops_per_batch]
            image_ids = image_id_per_box[i * crops_per_batch:(i + 1) * crops_per_batch]
            K_batch = Ks[i * crops_per_batch:(i + 1) * crops_per_batch]
            poses = self.predict_single_batch(
                image, K_batch, box_batch, image_ids, n_aug, rotmat, maybe_flip, flips, scales,
                gammas)
            result_batches = result_batches.write(i, poses)

        results_flat = result_batches.concat()
        return tf.RaggedTensor.from_row_lengths(results_flat, n_box_per_image)

    def predict_single_batch(
            self, images, K, boxes, image_ids, n_aug, rotmat, maybe_flip, flips, scales, gammas):
        n_box = tf.shape(boxes)[0]
        center_points = boxes[:, :2] + boxes[:, 2:4] / 2
        box_center_camspace = transf(center_points - K[:, :2, 2], tf.linalg.inv(K[:, :2, :2]))
        box_center_camspace = tf.concat(
            [box_center_camspace, tf.ones_like(box_center_camspace[:, :1])], axis=1)

        new_z = box_center_camspace / tf.linalg.norm(box_center_camspace, axis=-1, keepdims=True)
        new_x = tf.stack([new_z[:, 2], tf.zeros_like(new_z[:, 2]), -new_z[:, 0]], axis=1)
        new_y = tf.linalg.cross(new_z, new_x)
        nonaug_R = tf.stack([new_x, new_y, new_z], axis=1)
        new_R = maybe_flip[:, np.newaxis] @ rotmat[:, np.newaxis] @ nonaug_R
        box_scales = self.crop_side / tf.reduce_max(boxes[:, 2:4], axis=-1)
        new_K_mid = (tf.reshape(scales, [-1, 1, 1, 1]) *
                     tf.reshape(box_scales, [1, -1, 1, 1]) *
                     tf.reshape(K[:, :2, :2], [1, -1, 2, 2]))
        intrinsic_matrix = tf.concat([
            tf.concat([new_K_mid, tf.fill((n_aug, n_box, 2, 1), self.crop_side / 2)], axis=3),
            tf.concat([tf.zeros((n_aug, n_box, 1, 2), tf.float32),
                       tf.ones((n_aug, n_box, 1, 1), tf.float32)], axis=3)], axis=2)
        new_proj_matrix = intrinsic_matrix @ new_R
        homography = K @ tf.linalg.inv(new_proj_matrix)
        intrinsic_matrix_flat = tf.reshape(intrinsic_matrix, [n_aug * n_box, 3, 3])
        homography = tf.reshape(homography, [n_aug, n_box, 3, 3])
        homography = tf.reshape(homography, [-1, 9])
        homography = homography[:, :8] / homography[:, 8:]

        if self.antialias_factor > 1:
            H = homography
            a = self.antialias_factor
            homography = tf.stack([H[:, 0] / a, H[:, 1] / a, H[:, 2] - (a - 1) / 2,
                                   H[:, 3] / a, H[:, 4] / a, H[:, 5] - (a - 1) / 2,
                                   H[:, 6] / a, H[:, 7] / a], axis=1)

        temp_side = self.crop_side * self.antialias_factor
        image_ids = tf.tile(image_ids, [n_aug])
        crops = perspective_transform(
            images, homography, (temp_side, temp_side), 'BILINEAR', image_ids)

        crops = tf.cast(crops, tf.float16) / 255

        if self.antialias_factor > 1:
            crops = tf.image.resize(
                crops, (self.crop_side, self.crop_side), method=tf.image.ResizeMethod.AREA,
                antialias=True)
        crops = tf.reshape(crops, [n_aug, n_box * self.crop_side, self.crop_side, 3])
        crops **= tf.reshape(gammas, [-1, 1, 1, 1])
        crops = tf.reshape(crops, [-1, self.crop_side, self.crop_side, 3])

        poses = self.crop_model(crops, intrinsic_matrix_flat)
        poses = tf.reshape(poses, [n_aug, -1, tf.shape(poses)[1], 3])
        left_right_swapped = tf.gather(poses, self.joint_info.mirror_mapping, axis=2)
        poses = tf.where(tf.reshape(flips, [-1, 1, 1, 1]), left_right_swapped, poses)
        poses_origspace = tf.einsum('...nk,...jk->...nj', poses, tf.linalg.inv(new_R))
        return tf.reduce_mean(poses_origspace, axis=0)


def perspective_transform(images, homographies, output_shape, interpolation, image_ids):
    n_crops = tf.cast(tf.shape(homographies)[0], tf.int32)
    result_crops = tf.TensorArray(
        tf.uint8, size=n_crops, element_shape=(*output_shape, 3), infer_shape=False)
    for i in tf.range(n_crops):
        crop = tfa.image.transform(
            images[image_ids[i]], homographies[i], interpolation, output_shape)
        result_crops = result_crops.write(i, crop)
    return result_crops.stack()


def rotation_mat_zaxis(angle):
    zero = tf.zeros_like(angle)
    one = tf.ones_like(angle)
    sin = tf.math.sin(angle)
    cos = tf.math.cos(angle)
    return tf.stack([
        tf.stack([cos, -sin, zero], axis=-1),
        tf.stack([sin, cos, zero], axis=-1),
        tf.stack([zero, zero, one], axis=-1)], axis=-2)


def linspace_noend(start, stop, num):
    # Like np.linspace(endpoint=False)

    start = tf.convert_to_tensor(start)
    stop = tf.convert_to_tensor(stop, dtype=start.dtype)

    if num > 1:
        step = (stop - start) / tf.cast(num, start.dtype)
        new_stop = stop - step
        return tf.linspace(start, new_stop, num)
    else:
        return tf.linspace(start, stop, num)


def transf(points, matrices):
    return tf.einsum('...k,...jk->...j', points, matrices)


if __name__ == '__main__':
    main()
