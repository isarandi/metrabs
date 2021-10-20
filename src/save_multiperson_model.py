#!/usr/bin/env python3
import argparse

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import data.datasets3d
import options
import util
from options import FLAGS, logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-model-path', type=str, required=True)
    parser.add_argument('--output-model-path', type=str, required=True)
    parser.add_argument('--detector-path', type=str)
    parser.add_argument('--bone-length-dataset', type=str)
    parser.add_argument('--rot-aug', type=float, default=25)
    parser.add_argument('--rot-aug-linspace-noend', action=options.BoolAction)
    parser.add_argument('--crop-side', type=int, default=256)
    parser.add_argument('--detector-flip-vertical-too', action=options.BoolAction)
    options.initialize(parser)
    pose_estimator = Pose3dEstimator()
    tf.saved_model.save(
        pose_estimator, FLAGS.output_model_path,
        options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))
    logger.info(f'Full image model has been exported to {FLAGS.output_model_path}')


# Dummy value which will mean that the intrinsic_matrix are unknown
UNKNOWN_INTRINSIC_MATRIX = ((-1, -1, -1), (-1, -1, -1), (-1, -1, -1))
DEFAULT_EXTRINSIC_MATRIX = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
DEFAULT_DISTORTION = (0, 0, 0, 0, 0)
DEFAULT_WORLD_UP = (0, -1, 0)


class Pose3dEstimator(tf.Module):
    def __init__(self):
        super().__init__()

        # Note that only the Trackable resource attributes such as Variables and Models will be
        # retained when saving to SavedModel
        self.crop_model = tf.saved_model.load(FLAGS.input_model_path)
        self.crop_side = FLAGS.crop_side
        self.joint_names = self.crop_model.joint_names
        self.joint_edges = self.crop_model.joint_edges
        joint_names = [b.decode('utf8') for b in self.joint_names.numpy()]
        self.joint_info = data.datasets3d.JointInfo(joint_names, self.joint_edges.numpy())
        self.detector = tf.saved_model.load(FLAGS.detector_path) if FLAGS.detector_path else None

        if len(joint_names) == 122:
            skeleton_infos = util.load_pickle('./saved_model_export/skeleton_types.pkl')
            self.per_skeleton_indices = {
                k: tf.Variable(v['indices'], dtype=tf.int32, trainable=False)
                for k, v in skeleton_infos.items()}

            self.per_skeleton_joint_names = {
                k: tf.Variable(v['names'], dtype=tf.string, trainable=False)
                for k, v in skeleton_infos.items()}
            self.per_skeleton_joint_edges = {
                k: tf.Variable(v['edges'], dtype=tf.int32, trainable=False)
                for k, v in skeleton_infos.items()}
            self.per_skeleton_indices[''] = tf.range(122, dtype=tf.int32)
            self.per_skeleton_joint_names[''] = self.joint_names
            self.per_skeleton_joint_edges[''] = self.joint_edges

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.uint8),  # images
        tf.TensorSpec(shape=(None, 3, 3), dtype=tf.float32),  # intrinsic_matrix
        tf.TensorSpec(shape=(None, 5), dtype=tf.float32),  # distortion_coeffs
        tf.TensorSpec(shape=(None, 4, 4), dtype=tf.float32),  # extrinsic_matrix
        tf.TensorSpec(shape=(3,), dtype=tf.float32),  # world_up_vector
        tf.TensorSpec(shape=(), dtype=tf.float32),  # default_fov_degrees
        tf.TensorSpec(shape=(), dtype=tf.int32),  # internal_batch_size
        tf.TensorSpec(shape=(), dtype=tf.int32),  # antialias_factor
        tf.TensorSpec(shape=(), dtype=tf.int32),  # num_aug
        tf.TensorSpec(shape=(), dtype=tf.bool),  # average_aug
        tf.TensorSpec(shape=(), dtype=tf.string),  # skeleton
        tf.TensorSpec(shape=(), dtype=tf.float32),  # detector_threshold
        tf.TensorSpec(shape=(), dtype=tf.float32),  # detector_nms_iou_threshold
        tf.TensorSpec(shape=(), dtype=tf.int32),  # max_detections
        tf.TensorSpec(shape=(), dtype=tf.bool),  # detector_flip_aug
        tf.TensorSpec(shape=(), dtype=tf.bool)])  # suppress_implausible_poses
    def detect_poses_batched(
            self, images, intrinsic_matrix=(UNKNOWN_INTRINSIC_MATRIX,),
            distortion_coeffs=(DEFAULT_DISTORTION,), extrinsic_matrix=(DEFAULT_EXTRINSIC_MATRIX,),
            world_up_vector=DEFAULT_WORLD_UP, default_fov_degrees=55, internal_batch_size=64,
            antialias_factor=1, num_aug=5, average_aug=True, skeleton='', detector_threshold=0.3,
            detector_nms_iou_threshold=0.7, max_detections=-1, detector_flip_aug=False,
            suppress_implausible_poses=True):
        boxes = self._get_boxes(
            images, detector_flip_aug, detector_nms_iou_threshold, detector_threshold,
            max_detections)
        return self._estimate_poses_batched(
            images, boxes, intrinsic_matrix, distortion_coeffs, extrinsic_matrix, world_up_vector,
            default_fov_degrees, internal_batch_size, antialias_factor, num_aug, average_aug,
            skeleton, suppress_implausible_poses)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.uint8),  # images
        tf.RaggedTensorSpec(shape=(None, None, 4), ragged_rank=1, dtype=tf.float32),  # boxes
        tf.TensorSpec(shape=(None, 3, 3), dtype=tf.float32),  # intrinsic_matrix
        tf.TensorSpec(shape=(None, 5), dtype=tf.float32),  # distortion_coeffs
        tf.TensorSpec(shape=(None, 4, 4), dtype=tf.float32),  # extrinsic_matrix
        tf.TensorSpec(shape=(3,), dtype=tf.float32),  # world_up_vector
        tf.TensorSpec(shape=(), dtype=tf.float32),  # default_fov_degrees
        tf.TensorSpec(shape=(), dtype=tf.int32),  # internal_batch_size
        tf.TensorSpec(shape=(), dtype=tf.int32),  # antialias_factor
        tf.TensorSpec(shape=(), dtype=tf.int32),  # num_aug
        tf.TensorSpec(shape=(), dtype=tf.bool),  # average_aug
        tf.TensorSpec(shape=(), dtype=tf.string),  # skeleton
    ])
    def estimate_poses_batched(
            self, images, boxes, intrinsic_matrix=(UNKNOWN_INTRINSIC_MATRIX,),
            distortion_coeffs=(DEFAULT_DISTORTION,),
            extrinsic_matrix=(DEFAULT_EXTRINSIC_MATRIX,), world_up_vector=DEFAULT_WORLD_UP,
            default_fov_degrees=55, internal_batch_size=64, antialias_factor=1, num_aug=5,
            average_aug=True, skeleton=''):
        boxes = tf.concat([boxes, tf.ones_like(boxes[..., :1])], axis=-1)
        pred = self._estimate_poses_batched(
            images, boxes, intrinsic_matrix, distortion_coeffs, extrinsic_matrix, world_up_vector,
            default_fov_degrees, internal_batch_size, antialias_factor, num_aug, average_aug,
            skeleton, suppress_implausible_poses=False)
        del pred['boxes']
        return pred

    def _estimate_poses_batched(
            self, images, boxes, intrinsic_matrix, distortion_coeffs, extrinsic_matrix,
            world_up_vector, default_fov_degrees, internal_batch_size, antialias_factor, num_aug,
            average_aug, skeleton, suppress_implausible_poses):
        # Special case when zero boxes are provided or found
        # (i.e., all images images without person detections)
        # This must be explicitly handled, else the shapes don't work out automatically
        # for the TensorArray in _predict_in_batches.
        if tf.size(boxes) == 0:
            return self._predict_empty(images, num_aug, average_aug)

        n_images = tf.shape(images)[0]
        # If one intrinsic matrix is given, repeat it for all images
        if tf.shape(intrinsic_matrix)[0] == 1:
            # If intrinsic_matrix is not given, fill it in based on field of view
            if tf.reduce_all(intrinsic_matrix == -1):
                intrinsic_matrix = intrinsic_matrix_from_field_of_view(
                    default_fov_degrees, tf.shape(images)[1:3])
            intrinsic_matrix = tf.repeat(intrinsic_matrix, n_images, axis=0)

        # If one distortion coeff/extrinsic matrix is given, repeat it for all images
        if tf.shape(distortion_coeffs)[0] == 1:
            distortion_coeffs = tf.repeat(distortion_coeffs, n_images, axis=0)
        if tf.shape(extrinsic_matrix)[0] == 1:
            extrinsic_matrix = tf.repeat(extrinsic_matrix, n_images, axis=0)

        # Now repeat these camera params for each box
        n_box_per_image = boxes.row_lengths()
        intrinsic_matrix = tf.repeat(intrinsic_matrix, n_box_per_image, axis=0)
        distortion_coeffs = tf.repeat(distortion_coeffs, n_box_per_image, axis=0)

        # Up-vector in camera-space
        camspace_up = tf.einsum('c,bCc->bC', world_up_vector, extrinsic_matrix[..., :3, :3])
        camspace_up = tf.repeat(camspace_up, n_box_per_image, axis=0)

        # Set up the test-time augmentation parameters
        aug_gammas = tf.cast(tf.linspace(0.6, 1.0, num_aug), tf.float32)
        aug_angle_range = np.float32(np.deg2rad(FLAGS.rot_aug))
        if FLAGS.rot_aug_linspace_noend:
            aug_angles = linspace_noend(-aug_angle_range, aug_angle_range, num_aug)
        else:
            aug_angles = tf.linspace(-aug_angle_range, aug_angle_range, num_aug)
        aug_scales = tf.concat([
            linspace_noend(0.8, 1.0, num_aug // 2),
            tf.linspace(1.0, 1.1, num_aug - num_aug // 2)], axis=0)
        aug_should_flip = (tf.range(num_aug) - num_aug // 2) % 2 != 0
        aug_flipmat = tf.constant([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
        aug_maybe_flipmat = tf.where(
            aug_should_flip[:, np.newaxis, np.newaxis], aug_flipmat, tf.eye(3))
        aug_rotmat = rotation_mat_zaxis(-aug_angles)
        aug_rotflipmat = aug_maybe_flipmat @ aug_rotmat

        # crops_flat, poses3dcam_flat = self._predict_in_batches(
        poses3d_flat = self._predict_in_batches(
            images, intrinsic_matrix, distortion_coeffs, camspace_up, boxes, internal_batch_size,
            aug_should_flip, aug_rotflipmat, aug_gammas, aug_scales, antialias_factor)

        # Project the 3D poses to get the 2D poses
        poses2d_flat_normalized = to_homogeneous(
            distort_points(project(poses3d_flat), distortion_coeffs))
        poses2d_flat = tf.einsum('bank,bjk->banj', poses2d_flat_normalized,
                                 intrinsic_matrix[..., :2, :])
        poses2d_flat = tf.ensure_shape(poses2d_flat, [None, None, self.joint_info.n_joints, 2])

        # Arrange the results back into ragged tensors
        poses3d = tf.RaggedTensor.from_row_lengths(poses3d_flat, n_box_per_image)
        poses2d = tf.RaggedTensor.from_row_lengths(poses2d_flat, n_box_per_image)
        # crops = tf.RaggedTensor.from_row_lengths(crops_flat, n_box_per_image)

        if suppress_implausible_poses:
            # Filter the resulting poses for individual plausibility to reduce false positives
            selected_indices = self._filter_poses(boxes, poses3d, poses2d)
            boxes, poses3d, poses2d = [
                tf.gather(x, selected_indices, batch_dims=1)
                for x in [boxes, poses3d, poses2d]]
            # crops = tf.gather(crops, selected_indices, batch_dims=1)

        # Convert to world coordinates
        extrinsic_matrix = tf.repeat(tf.linalg.inv(extrinsic_matrix), poses3d.row_lengths(), axis=0)
        poses3d = tf.RaggedTensor.from_row_lengths(
            tf.einsum(
                'bank,bjk->banj', to_homogeneous(poses3d.flat_values),
                extrinsic_matrix[..., :3, :]),
            poses3d.row_lengths())

        if skeleton != '':
            poses3d = self._get_skeleton(poses3d, skeleton)
            poses2d = self._get_skeleton(poses2d, skeleton)

        if average_aug:
            poses3d = tf.reduce_mean(poses3d, axis=-3)
            poses2d = tf.reduce_mean(poses2d, axis=-3)

        result = dict(boxes=boxes, poses3d=poses3d, poses2d=poses2d)
        # result['crops'] = crops
        return result

    def _get_boxes(
            self, images, detector_flip_aug, detector_nms_iou_threshold, detector_threshold,
            max_detections):
        if self.detector is None:
            n_images = tf.shape(images)[0]
            boxes = tf.RaggedTensor.from_row_lengths(
                tf.zeros(shape=(0, 5)), tf.zeros(n_images, tf.int64))
        else:
            boxes = self.detector.predict_multi_image(
                images, detector_threshold, detector_nms_iou_threshold, detector_flip_aug,
                detector_flip_aug and FLAGS.detector_flip_vertical_too)
            if max_detections > -1 and not tf.size(boxes) == 0:
                topk_indices = topk_indices_ragged(boxes[..., 4], max_detections)
                boxes = tf.gather(boxes, topk_indices, axis=1, batch_dims=1)
        return boxes

    def _predict_in_batches(
            self, images, intrinsic_matrix, distortion_coeffs, camspace_up, boxes,
            internal_batch_size, aug_should_flip, aug_rotflipmat, aug_gammas, aug_scales,
            antialias_factor):

        num_aug = tf.shape(aug_gammas)[0]
        boxes_per_batch = internal_batch_size // num_aug
        boxes_flat = boxes.flat_values
        image_id_per_box = boxes.value_rowids()

        # Gamma decoding for correct image rescaling later on
        images = (tf.cast(images, tf.float32) / np.float32(255)) ** 2.2

        if boxes_per_batch == 0:
            # Run all as a single batch
            return self._predict_single_batch(
                images, intrinsic_matrix, distortion_coeffs, camspace_up, boxes_flat,
                image_id_per_box, aug_rotflipmat, aug_should_flip, aug_scales, aug_gammas,
                antialias_factor)
        else:
            # Chunk the image crops into batches and predict them one by one
            n_total_boxes = tf.shape(boxes_flat)[0]
            n_batches = tf.cast(tf.math.ceil(n_total_boxes / boxes_per_batch), tf.int32)
            poses3d_batches = tf.TensorArray(
                tf.float32, size=n_batches, element_shape=(None, None, self.joint_info.n_joints, 3),
                infer_shape=False)
            # crop_batches = tf.TensorArray(
            #     tf.float32, size=n_batches,
            #     element_shape=(None, None, self.crop_side, self.crop_side, 3),
            #     infer_shape=False)

            for i in tf.range(n_batches):
                batch_slice = slice(i * boxes_per_batch, (i + 1) * boxes_per_batch)
                # crops, poses3d = self._predict_single_batch(
                poses3d = self._predict_single_batch(
                    images, intrinsic_matrix[batch_slice], distortion_coeffs[batch_slice],
                    camspace_up[batch_slice], boxes_flat[batch_slice],
                    image_id_per_box[batch_slice], aug_rotflipmat, aug_should_flip, aug_scales,
                    aug_gammas, antialias_factor)
                poses3d_batches = poses3d_batches.write(i, poses3d)
                # crop_batches = crop_batches.write(i, crops)

            # return crop_batches.concat(), poses3d_batches.concat()
            return poses3d_batches.concat()

    def _predict_single_batch(
            self, images, intrinsic_matrix, distortion_coeffs, camspace_up, boxes, image_ids,
            aug_rotflipmat, aug_should_flip, aug_scales, aug_gammas, antialias_factor):
        # Get crops and info about the transformation used to create them
        # Each has shape [num_aug, n_boxes, ...]
        crops, new_intrinsic_matrix, R = self._get_crops(
            images, intrinsic_matrix, distortion_coeffs, camspace_up, boxes, image_ids,
            aug_rotflipmat, aug_scales, aug_gammas, antialias_factor)

        # Flatten each and predict the pose with the crop model
        new_intrinsic_matrix_flat = tf.reshape(new_intrinsic_matrix, (-1, 3, 3))
        crops_flat = tf.reshape(crops, (-1, self.crop_side, self.crop_side, 3))
        poses_flat = self.crop_model.predict_multi(
            tf.cast(crops_flat, tf.float16),
            new_intrinsic_matrix_flat)

        # Unflatten the result
        num_aug = tf.shape(aug_should_flip)[0]
        poses = tf.reshape(poses_flat, [num_aug, -1, self.joint_info.n_joints, 3])

        # Reorder the joints for mirrored predictions (e.g., swap left and right wrist)
        left_right_swapped_poses = tf.gather(poses, self.joint_info.mirror_mapping, axis=-2)
        poses = tf.where(
            tf.reshape(aug_should_flip, [-1, 1, 1, 1]), left_right_swapped_poses, poses)

        # Transform the predictions back into the original camera space
        # We need to multiply by the inverse of R, but since we are using row vectors in `poses`
        # the inverse and transpose cancel out, leaving just R.
        poses_orig_camspace = poses @ R

        # Transpose to [n_boxes, num_aug, ...]
        # return (tf.transpose(crops, [1, 0, 2, 3, 4]),
        # tf.transpose(poses_orig_camspace, [1, 0, 2, 3]))
        return tf.transpose(poses_orig_camspace, [1, 0, 2, 3])

    def _get_crops(
            self, images, intrinsic_matrix, distortion_coeffs, camspace_up, boxes, image_ids,
            aug_rotflipmat, aug_scales, aug_gammas, antialias_factor):
        R_noaug, box_aug_scales = self._get_new_rotation_and_scale(
            intrinsic_matrix, distortion_coeffs, camspace_up, boxes)

        # How much we need to scale overall, taking scale augmentation into account
        # From here on, we introduce the dimension of augmentations
        crop_scales = aug_scales[:, tf.newaxis] * box_aug_scales[tf.newaxis, :]
        # Build the new intrinsic matrix
        n_box = tf.shape(boxes)[0]
        num_aug = tf.shape(aug_gammas)[0]
        new_intrinsic_matrix = tf.concat([
            tf.concat([
                # Top-left of original intrinsic matrix gets scaled
                intrinsic_matrix[tf.newaxis, :, :2, :2] * crop_scales[:, :, tf.newaxis, tf.newaxis],
                # Principal point is the middle of the new image size
                tf.fill((num_aug, n_box, 2, 1), self.crop_side / 2)], axis=3),
            tf.concat([
                # [0, 0, 1] as the last row of the intrinsic matrix:
                tf.zeros((num_aug, n_box, 1, 2), tf.float32),
                tf.ones((num_aug, n_box, 1, 1), tf.float32)], axis=3)], axis=2)
        R = aug_rotflipmat[:, tf.newaxis] @ R_noaug
        new_invprojmat = tf.linalg.inv(new_intrinsic_matrix @ R)

        # If we perform antialiasing through output scaling, we render a larger image first and then
        # shrink it. So we scale the homography first.
        if antialias_factor > 1:
            scaling_mat = corner_aligned_scale_mat(1 / tf.cast(antialias_factor, tf.float32))
            new_invprojmat = new_invprojmat @ scaling_mat

        crops = warp_images(
            images,
            intrinsic_matrix=tf.tile(intrinsic_matrix, [num_aug, 1, 1]),
            new_invprojmat=tf.reshape(new_invprojmat, [-1, 3, 3]),
            distortion_coeffs=tf.tile(distortion_coeffs, [num_aug, 1]),
            crop_scales=tf.reshape(crop_scales, [-1]) * tf.cast(antialias_factor, tf.float32),
            output_shape=(self.crop_side * antialias_factor, self.crop_side * antialias_factor),
            image_ids=tf.tile(image_ids, [num_aug]))

        # Downscale the result if we do antialiasing through output scaling
        if antialias_factor == 2:
            crops = tf.nn.avg_pool2d(crops, 2, 2, padding='VALID')
        elif antialias_factor == 4:
            crops = tf.nn.avg_pool2d(crops, 4, 4, padding='VALID')
        elif antialias_factor > 4:
            crops = tf.image.resize(
                crops, (self.crop_side, self.crop_side), tf.image.ResizeMethod.AREA)
        crops = tf.reshape(crops, [num_aug, n_box, self.crop_side, self.crop_side, 3])
        # The division by 2.2 cancels the original gamma decoding from earlier
        crops **= tf.reshape(aug_gammas / 2.2, [-1, 1, 1, 1, 1])
        return crops, new_intrinsic_matrix, R

    def _get_new_rotation_and_scale(self, intrinsic_matrix, distortion_coeffs, camspace_up, boxes):
        # Transform five points on each box: the center and the four side midpoints
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxpoints_homog = to_homogeneous(tf.stack([
            tf.stack([x + w / 2, y + h / 2], axis=1),
            tf.stack([x + w / 2, y], axis=1),
            tf.stack([x + w, y + h / 2], axis=1),
            tf.stack([x + w / 2, y + h], axis=1),
            tf.stack([x, y + h / 2], axis=1)], axis=1))
        boxpoints_camspace = tf.einsum(
            'bpc,bCc->bpC', boxpoints_homog, tf.linalg.inv(intrinsic_matrix))
        boxpoints_camspace = to_homogeneous(
            undistort_points(boxpoints_camspace[..., :2], distortion_coeffs))

        # Create a rotation matrix that will put the box center to the principal point
        # and apply the augmentation rotation and flip, to get the new coordinate frame
        box_center_camspace = boxpoints_camspace[:, 0]
        R_noaug = get_new_rotation_matrix(forward_vector=box_center_camspace, up_vector=camspace_up)

        # Transform the side midpoints of the box to the new coordinate frame
        sidepoints_camspace = boxpoints_camspace[:, 1:5]
        sidepoints_new = project(tf.einsum(
            'bpc,bCc->bpC', sidepoints_camspace, intrinsic_matrix @ R_noaug))

        # Measure the size of the reprojected boxes
        vertical_size = tf.linalg.norm(sidepoints_new[:, 0] - sidepoints_new[:, 2], axis=-1)
        horiz_size = tf.linalg.norm(sidepoints_new[:, 1] - sidepoints_new[:, 3], axis=-1)
        box_size_new = tf.maximum(vertical_size, horiz_size)

        # How much we need to scale (zoom) to have the boxes fill out the final crop
        box_aug_scales = self.crop_side / box_size_new
        return R_noaug, box_aug_scales

    def _predict_empty(self, image, num_aug, average_aug):
        if average_aug:
            poses3d = tf.zeros(shape=(0, self.joint_info.n_joints, 3))
            poses2d = tf.zeros(shape=(0, self.joint_info.n_joints, 2))
        else:
            poses3d = tf.zeros(shape=(0, num_aug, self.joint_info.n_joints, 3))
            poses2d = tf.zeros(shape=(0, num_aug, self.joint_info.n_joints, 2))

        n_images = tf.shape(image)[0]
        poses3d = tf.RaggedTensor.from_row_lengths(poses3d, tf.zeros(n_images, tf.int64))
        poses2d = tf.RaggedTensor.from_row_lengths(poses2d, tf.zeros(n_images, tf.int64))
        boxes = tf.zeros(shape=(0, 5))
        boxes = tf.RaggedTensor.from_row_lengths(boxes, tf.zeros(n_images, tf.int64))

        result = dict(boxes=boxes, poses3d=poses3d, poses2d=poses2d)
        #     crops = tf.zeros(shape=(0, num_aug, self.crop_side, self.crop_side, 3))
        #     crops = tf.RaggedTensor.from_row_lengths(crops, tf.zeros(n_images, tf.int64))
        #     result['crops'] = crops
        return result

    def _filter_poses(self, boxes, poses3d, poses2d):
        poses3d_mean = tf.reduce_mean(poses3d, axis=-3)
        poses2d_mean = tf.reduce_mean(poses2d, axis=-3)

        plausible_mask_flat = tf.logical_and(
            tf.logical_and(
                is_pose_plausible(poses3d_mean.flat_values),
                are_augmentation_results_consistent(poses3d.flat_values)),
            is_pose_consistent_with_box(poses2d_mean.flat_values, boxes.flat_values))

        plausible_mask = tf.RaggedTensor.from_row_lengths(
            plausible_mask_flat, boxes.row_lengths())

        # Apply pose similarity-based non-maximum suppression to reduce duplicates
        nms_indices = tf.map_fn(
            fn=lambda args: pose_non_max_suppression(*args),
            elems=(poses3d_mean, boxes[..., 4], plausible_mask),
            fn_output_signature=tf.RaggedTensorSpec(shape=(None,), dtype=tf.int32))
        return nms_indices

    def _get_skeleton(self, poses, skeleton):
        # We must list all possibilities since we can't address the Python dictionary
        # `self.per_skeleton_indices` with the tf.Tensor `skeleton`.
        if skeleton == b'smpl_24':
            indices = self.per_skeleton_indices['smpl_24']
        elif skeleton == b'coco_19':
            indices = self.per_skeleton_indices['coco_19']
        elif skeleton == b'h36m_17':
            indices = self.per_skeleton_indices['h36m_17']
        elif skeleton == b'h36m_25':
            indices = self.per_skeleton_indices['h36m_25']
        elif skeleton == b'mpi_inf_3dhp_17':
            indices = self.per_skeleton_indices['mpi_inf_3dhp_17']
        elif skeleton == b'mpi_inf_3dhp_28':
            indices = self.per_skeleton_indices['mpi_inf_3dhp_28']
        elif skeleton == b'smpl+head_30':
            indices = self.per_skeleton_indices['smpl+head_30']
        else:
            indices = tf.range(122, dtype=tf.int32)

        return tf.gather(poses, indices, axis=-2)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),  # image
        tf.TensorSpec(shape=(3, 3), dtype=tf.float32),  # intrinsic_matrix
        tf.TensorSpec(shape=(5,), dtype=tf.float32),  # distortion_coeffs
        tf.TensorSpec(shape=(4, 4), dtype=tf.float32),  # extrinsic_matrix
        tf.TensorSpec(shape=(3,), dtype=tf.float32),  # world_up_vector
        tf.TensorSpec(shape=(), dtype=tf.float32),  # default_fov_degrees
        tf.TensorSpec(shape=(), dtype=tf.int32),  # internal_batch_size
        tf.TensorSpec(shape=(), dtype=tf.int32),  # antialias_factor
        tf.TensorSpec(shape=(), dtype=tf.int32),  # num_aug
        tf.TensorSpec(shape=(), dtype=tf.bool),  # average_aug
        tf.TensorSpec(shape=(), dtype=tf.string),  # skeleton
        tf.TensorSpec(shape=(), dtype=tf.float32),  # detector_threshold
        tf.TensorSpec(shape=(), dtype=tf.float32),  # detector_nms_iou_threshold
        tf.TensorSpec(shape=(), dtype=tf.int32),  # max_detections
        tf.TensorSpec(shape=(), dtype=tf.bool),  # detector_flip_aug
        tf.TensorSpec(shape=(), dtype=tf.bool)  # suppress_implausible_poses
    ])
    def detect_poses(
            self, image, intrinsic_matrix=UNKNOWN_INTRINSIC_MATRIX,
            distortion_coeffs=DEFAULT_DISTORTION, extrinsic_matrix=DEFAULT_EXTRINSIC_MATRIX,
            world_up_vector=DEFAULT_WORLD_UP, default_fov_degrees=55, internal_batch_size=64,
            antialias_factor=1, num_aug=5, average_aug=True, skeleton='', detector_threshold=0.3,
            detector_nms_iou_threshold=0.7, max_detections=-1, detector_flip_aug=False,
            suppress_implausible_poses=True):
        images = image[tf.newaxis]
        intrinsic_matrix = intrinsic_matrix[tf.newaxis]
        distortion_coeffs = distortion_coeffs[tf.newaxis]
        extrinsic_matrix = extrinsic_matrix[tf.newaxis]
        result = self.detect_poses_batched(
            images, intrinsic_matrix, distortion_coeffs, extrinsic_matrix, world_up_vector,
            default_fov_degrees, internal_batch_size, antialias_factor, num_aug, average_aug,
            skeleton, detector_threshold, detector_nms_iou_threshold, max_detections,
            detector_flip_aug, suppress_implausible_poses)
        return tf.nest.map_structure(lambda x: tf.squeeze(x, 0), result)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),  # image
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),  # boxes
        tf.TensorSpec(shape=(3, 3), dtype=tf.float32),  # intrinsic_matrix
        tf.TensorSpec(shape=(5,), dtype=tf.float32),  # distortion_coeffs
        tf.TensorSpec(shape=(4, 4), dtype=tf.float32),  # extrinsic_matrix
        tf.TensorSpec(shape=(3,), dtype=tf.float32),  # world_up_vector
        tf.TensorSpec(shape=(), dtype=tf.float32),  # default_fov_degrees
        tf.TensorSpec(shape=(), dtype=tf.int32),  # internal_batch_size
        tf.TensorSpec(shape=(), dtype=tf.int32),  # antialias_factor
        tf.TensorSpec(shape=(), dtype=tf.int32),  # num_aug
        tf.TensorSpec(shape=(), dtype=tf.bool),  # average_aug
        tf.TensorSpec(shape=(), dtype=tf.string),  # skeleton
    ])
    def estimate_poses(
            self, image, boxes, intrinsic_matrix=UNKNOWN_INTRINSIC_MATRIX,
            distortion_coeffs=DEFAULT_DISTORTION, extrinsic_matrix=DEFAULT_EXTRINSIC_MATRIX,
            world_up_vector=DEFAULT_WORLD_UP, default_fov_degrees=55, internal_batch_size=64,
            antialias_factor=1, num_aug=5, average_aug=True, skeleton=''):
        images = image[tf.newaxis]
        boxes = tf.RaggedTensor.from_tensor(boxes[tf.newaxis])
        intrinsic_matrix = intrinsic_matrix[tf.newaxis]
        distortion_coeffs = distortion_coeffs[tf.newaxis]
        extrinsic_matrix = extrinsic_matrix[tf.newaxis]
        result = self.estimate_poses_batched(
            images, boxes, intrinsic_matrix, distortion_coeffs, extrinsic_matrix, world_up_vector,
            default_fov_degrees, internal_batch_size, antialias_factor, num_aug, average_aug,
            skeleton)
        return tf.nest.map_structure(lambda x: tf.squeeze(x, 0), result)


def corner_aligned_scale_mat(factor):
    shift = (factor - 1) / 2
    return tf.convert_to_tensor(
        [[factor, 0, shift],
         [0, factor, shift],
         [0, 0, 1]], tf.float32)


def get_new_rotation_matrix(forward_vector, up_vector):
    # Z will point forwards, towards the box center
    new_z = forward_vector / tf.linalg.norm(forward_vector, axis=-1, keepdims=True)
    # Get the X (right direction) as the cross of forward and up.
    new_x = tf.linalg.cross(new_z, up_vector)
    # Get alternative X by rotating the new Z around the old Y by 90 degrees
    # in case lookdir happens to align with the up vector and the above cross product is zero.
    new_x_alt = tf.stack([new_z[:, 2], tf.zeros_like(new_z[:, 2]), -new_z[:, 0]], axis=1)
    new_x = tf.where(tf.linalg.norm(new_x, axis=-1, keepdims=True) == 0, new_x_alt, new_x)
    new_x = new_x / tf.linalg.norm(new_x, axis=-1, keepdims=True)
    # Complete the right-handed coordinate system to get Y
    new_y = tf.linalg.cross(new_z, new_x)
    # Stack the axis vectors to get the rotation matrix
    return tf.stack([new_x, new_y, new_z], axis=1)


def intrinsic_matrix_from_field_of_view(fov_degrees, imshape):
    imshape = tf.cast(imshape, tf.float32)
    fov_radians = fov_degrees * tf.constant(np.pi / 180, tf.float32)
    larger_side = tf.reduce_max(imshape)
    focal_length = larger_side / (tf.math.tan(fov_radians / 2) * 2)
    return tf.convert_to_tensor(
        [[[focal_length, 0, imshape[1] / 2],
          [0, focal_length, imshape[0] / 2],
          [0, 0, 1]]], tf.float32)


def warp_images(
        images, intrinsic_matrix, new_invprojmat, distortion_coeffs, crop_scales, output_shape,
        image_ids):
    # Create a simple pyramid with lower resolution images for simple antialiasing.
    n_pyramid_levels = 3
    image_levels = [images]
    for _ in range(1, n_pyramid_levels):
        image_levels.append(tf.nn.avg_pool2d(image_levels[-1], 2, 2, padding='VALID'))

    intrinsic_matrix_levels = [
        corner_aligned_scale_mat(1 / 2 ** i_level) @ intrinsic_matrix
        for i_level in range(n_pyramid_levels)]

    # Decide which pyramid level is most appropriate for each crop
    i_pyramid_levels = tf.math.floor(-tf.math.log(crop_scales) / np.log(2))
    i_pyramid_levels = tf.cast(
        tf.clip_by_value(i_pyramid_levels, 0, n_pyramid_levels - 1), tf.int32)

    n_crops = tf.cast(tf.shape(new_invprojmat)[0], tf.int32)
    result_crops = tf.TensorArray(
        images.dtype, size=n_crops, element_shape=(None, None, 3), infer_shape=False)
    for i_crop in tf.range(n_crops):
        tf.autograph.experimental.set_loop_options(parallel_iterations=1000)

        i_pyramid_level = i_pyramid_levels[i_crop]
        # Ugly, but we must unroll this because we can't index a Python list with a Tensor...
        if i_pyramid_level == 0:
            image_level = image_levels[0]
            intrinsic_matrix_level = intrinsic_matrix_levels[0]
        elif i_pyramid_level == 1:
            image_level = image_levels[1]
            intrinsic_matrix_level = intrinsic_matrix_levels[1]
        else:
            image_level = image_levels[2]
            intrinsic_matrix_level = intrinsic_matrix_levels[2]

        # Perform the transformation on using the selected pyramid level
        crop = warp_single_image(
            image_level[image_ids[i_crop]], intrinsic_matrix_level[i_crop], new_invprojmat[i_crop],
            distortion_coeffs[i_crop], output_shape)
        result_crops = result_crops.write(i_crop, crop)
    return result_crops.stack()


def warp_single_image(image, intrinsic_matrix, new_invprojmat, distortion_coeffs, output_shape):
    if tf.reduce_all(distortion_coeffs == 0):
        # No lens distortion, simply apply a homography
        H = intrinsic_matrix @ new_invprojmat
        H = tf.reshape(H, [9])[:8] / H[2, 2]
        return tfa.image.transform(image, H, output_shape=output_shape)
    else:
        # With lens distortion, we must transform each pixel and interpolate
        new_coords = tf.cast(tf.stack(tf.meshgrid(
            tf.range(output_shape[1]), tf.range(output_shape[0])), axis=-1), tf.float32)
        new_coords_homog = to_homogeneous(new_coords)
        old_coords_homog = tf.einsum('hwc,Cc->hwC', new_coords_homog, new_invprojmat)
        old_coords_homog = to_homogeneous(
            distort_points(project(old_coords_homog), distortion_coeffs))
        old_coords = tf.einsum('hwc,Cc->hwC', old_coords_homog, intrinsic_matrix)[..., :2]
        interpolated = tfa.image.interpolate_bilinear(
            image[tf.newaxis], tf.reshape(old_coords, [1, -1, 2])[..., ::-1])
        return tf.reshape(interpolated, [output_shape[0], output_shape[1], 3])


def project(points):
    return points[..., :2] / points[..., 2:3]


def distort_points(undist_points2d, distortion_coeffs):
    if tf.reduce_all(distortion_coeffs == 0):
        return undist_points2d
    else:
        a, b, c = distortion_formula_parts(undist_points2d, distortion_coeffs)
        return undist_points2d * (a + b) + c


def undistort_points(dist_points2d, distortion_coeffs):
    if tf.reduce_all(distortion_coeffs == 0):
        return dist_points2d

    undist_points2d = dist_points2d
    for _ in range(5):
        a, b, c = distortion_formula_parts(undist_points2d, distortion_coeffs)
        undist_points2d = (dist_points2d - c - undist_points2d * b) / a
    return undist_points2d


def distortion_formula_parts(undist_points2d, distortion_coeffs):
    distortion_coeffs_broadcast_shape = (
            ([-1] if distortion_coeffs.shape.rank > 1 else []) +
            [1] * (undist_points2d.shape.rank - distortion_coeffs.shape.rank) + [5])
    distortion_coeffs = tf.reshape(distortion_coeffs, distortion_coeffs_broadcast_shape)

    r2 = tf.reduce_sum(tf.square(undist_points2d), axis=-1, keepdims=True)
    a = ((distortion_coeffs[..., 4:5] * r2 + distortion_coeffs[..., 1:2]) * r2 +
         distortion_coeffs[..., 0:1]) * r2 + 1
    b = 2 * tf.reduce_sum(undist_points2d * distortion_coeffs[..., 3:1:-1], axis=-1, keepdims=True)
    c = r2 * distortion_coeffs[..., 3:1:-1]
    return a, b, c


def to_homogeneous(x):
    return tf.concat([x, tf.ones_like(x[..., :1])], axis=-1)


def rotation_mat_zaxis(angle):
    sin = tf.math.sin(angle)
    cos = tf.math.cos(angle)
    _0 = tf.zeros_like(angle)
    _1 = tf.ones_like(angle)
    return tf.stack([
        tf.stack([cos, -sin, _0], axis=-1),
        tf.stack([sin, cos, _0], axis=-1),
        tf.stack([_0, _0, _1], axis=-1)], axis=-2)


def topk_indices_ragged(inp, k=1):
    row_lengths = inp.row_lengths()
    inp = inp.to_tensor(default_value=-np.inf)
    result = tf.math.top_k(inp, k=k, sorted=False).indices
    return tf.RaggedTensor.from_tensor(result, row_lengths)


def are_augmentation_results_consistent(poses3d):
    """At least one fourth of the joints have a standard deviation under 200 mm"""
    n_joints = poses3d.shape[-2]
    stdevs = point_stdev(scale_align(poses3d), item_axis=1, coord_axis=-1)
    return tf.math.count_nonzero(stdevs < 200, axis=1) > (n_joints // 4)


def scale_align(poses):
    square_scales = tf.reduce_mean(tf.square(poses), axis=(-2, -1), keepdims=True)
    mean_square_scale = tf.reduce_mean(square_scales, axis=-3, keepdims=True)
    return poses * tf.sqrt(mean_square_scale / square_scales)


def point_stdev(poses, item_axis, coord_axis):
    coordwise_variance = tf.math.reduce_variance(poses, axis=item_axis, keepdims=True)
    average_stdev = tf.sqrt(tf.reduce_sum(coordwise_variance, axis=coord_axis, keepdims=True))
    return tf.squeeze(average_stdev, (item_axis, coord_axis))


def pose_non_max_suppression(poses, scores, is_pose_valid):
    plausible_indices_single_frame = tf.squeeze(tf.where(is_pose_valid), 1)
    plausible_poses = tf.gather(poses, plausible_indices_single_frame)
    plausible_scores = tf.gather(scores, plausible_indices_single_frame)
    similarity_matrix = compute_pose_similarity(plausible_poses)
    nms_indices = tf.image.non_max_suppression_overlaps(
        overlaps=similarity_matrix, scores=plausible_scores,
        max_output_size=150, overlap_threshold=0.4)
    return tf.cast(tf.gather(plausible_indices_single_frame, nms_indices), tf.int32)


def compute_pose_similarity(poses):
    # Pairwise scale align the poses before comparing them
    square_scales = tf.reduce_mean(tf.square(poses), axis=(-2, -1), keepdims=True)
    square_scales1 = tf.expand_dims(square_scales, 0)
    square_scales2 = tf.expand_dims(square_scales, 1)
    mean_square_scales = (square_scales1 + square_scales2) / 2
    scale_factor1 = tf.sqrt(mean_square_scales / square_scales1)
    scale_factor2 = tf.sqrt(mean_square_scales / square_scales2)

    poses1 = tf.expand_dims(poses, 0)
    poses2 = tf.expand_dims(poses, 1)

    dists = tf.linalg.norm(scale_factor1 * poses1 - scale_factor2 * poses2, axis=-1)
    best_dists = tf.math.top_k(dists, k=poses.shape[-2] // 4, sorted=False).values
    return tf.reduce_mean(tf.nn.relu(1 - best_dists / 300), axis=-1)


def is_pose_plausible(poses):
    """Check poses for very basic plausibility, mainly to deal with false positive detections
    that don't yield a meaningful pose prediction.
    """
    bone_dataset = data.datasets3d.get_dataset(FLAGS.bone_length_dataset)
    ji = bone_dataset.joint_info
    bone_lengths = tf.stack([
        tf.norm(poses[:, i] - poses[:, j], axis=-1)
        for i, j in ji.stick_figure_edges], axis=-1)

    bone_length_relative = bone_lengths / bone_dataset.train_bones
    bone_length_diff = tf.abs(bone_lengths - bone_dataset.train_bones)

    relsmall = bone_length_relative < 0.1
    relbig = bone_length_relative > 3
    absdiffbig = bone_length_diff > 300
    is_implausible = tf.reduce_any(
        tf.logical_and(tf.logical_or(relbig, relsmall), absdiffbig), axis=-1)
    return tf.logical_not(is_implausible)


def is_pose_consistent_with_box(pose2d, box):
    """Check if pose prediction is consistent with the original box it was based on.
    Concretely, check if the intersection between the pose's bounding box and the detection has
    at least half the area of the detection box. This is like IoU but the denominator is the
    area of the detection box, so that truncated poses are handled correctly.
    """

    # Compute the bounding box around the 2D joints
    posebox_start = tf.reduce_min(pose2d, axis=-2)
    posebox_end = tf.reduce_max(pose2d, axis=-2)

    box_start = box[..., :2]
    box_end = box[..., :2] + box[..., 2:4]
    box_area = tf.reduce_prod(box[..., 2:4], axis=-1)

    intersection_start = tf.maximum(box_start, posebox_start)
    intersection_end = tf.minimum(box_end, posebox_end)
    intersection_area = tf.reduce_prod(tf.nn.relu(intersection_end - intersection_start), axis=-1)
    return intersection_area > 0.5 * box_area


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


if __name__ == '__main__':
    main()
