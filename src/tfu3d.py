import einops
import numpy as np
import tensorflow as tf
from tensorflow_graphics.math.optimizer.levenberg_marquardt import minimize as levenberg_marquardt

import tfu
from init import FLAGS


def rigid_align(
        coords_pred, coords_true, *, joint_validity_mask=None, scale_align=False,
        reflection_align=False):
    """Returns the predicted coordinates after rigid alignment to the ground truth."""
    if joint_validity_mask is None:
        joint_validity_mask = tf.ones_like(coords_pred[..., 0], dtype=tf.bool)

    return procrustes_tf(
        coords_true, coords_pred, joint_validity_mask, allow_scaling=scale_align,
        allow_reflection=reflection_align)


def center_relative_pose(
        coords3d, joint_validity_mask=None, center_is_mean=False, center_joints=None):
    if center_is_mean:
        if isinstance(coords3d, np.ndarray):
            if joint_validity_mask is None:
                center = np.mean(coords3d, axis=1, keepdims=True)
            else:
                coords3d = coords3d.copy()
                coords3d[~joint_validity_mask] = np.nan
                center = np.nanmean(coords3d, axis=1, keepdims=True)
        else:
            if joint_validity_mask is None:
                center = tf.reduce_mean(coords3d, axis=1, keepdims=True)
            else:
                if center_joints is not None:
                    center = tfu.reduce_mean_masked(
                        tf.gather(coords3d, center_joints, axis=1),
                        tf.gather(joint_validity_mask, center_joints, axis=1),
                        axis=1, keepdims=True)
                else:
                    center = tfu.reduce_mean_masked(
                        coords3d, joint_validity_mask, axis=1, keepdims=True)
    else:
        center = coords3d[:, -1:]
    return coords3d - center


def linear_combine_points(coords, weights):
    return tf.einsum('bjc,jJ->bJc', coords, weights)


def procrustes_tf(X, Y, validity_mask, allow_scaling=False, allow_reflection=False):
    """Register the points in Y by rotation, translation, uniform scaling (optional) and
    reflection (optional)
    to be closest to the corresponding points in X, in a least-squares sense.

    This function operates on batches. For each item in the batch a separate
    transform is computed independently of the others.

    Arguments:
       X: Tensor with shape [batch_size, n_points, point_dimensionality]
       Y: Tensor with shape [batch_size, n_points, point_dimensionality]
       validity_mask: Boolean Tensor with shape [batch_size, n_points] indicating
         whether a point is valid in Y
       allow_scaling: boolean, specifying whether uniform scaling is allowed
       allow_reflection: boolean, specifying whether reflections are allowed

    Returns the transformed version of Y.
    """

    validity_mask = validity_mask[..., np.newaxis]
    _0 = tf.constant(0, X.dtype)
    n_points_per_example = tf.math.count_nonzero(
        validity_mask, axis=1, dtype=tf.float32, keepdims=True)
    denominator_correction_factor = validity_mask.shape[1] / n_points_per_example

    def normalize(Z):
        Z = tf.where(validity_mask, Z, _0)
        mean = tf.reduce_mean(Z, axis=1, keepdims=True) * denominator_correction_factor
        centered = tf.where(validity_mask, Z - mean, _0)
        norm = tf.norm(centered, axis=(1, 2), ord='fro', keepdims=True)
        normalized = centered / norm
        return mean, norm, normalized

    meanX, normX, normalizedX = normalize(X)
    meanY, normY, normalizedY = normalize(Y)
    A = tf.matmul(normalizedY, normalizedX, transpose_a=True)
    s, U, V = tf.linalg.svd(A, full_matrices=False)
    T = tf.matmul(U, V, transpose_b=True)
    s = tf.expand_dims(s, axis=-1)

    if allow_scaling:
        relative_scale = normX / normY
        output_scale = relative_scale * tf.reduce_sum(s, axis=1, keepdims=True)
    else:
        relative_scale = None
        output_scale = 1

    if not allow_reflection:
        # Check if T has a reflection component. If so, then remove it by flipping
        # across the direction of least variance, i.e. the last singular value/vector.
        has_reflection = (tf.linalg.det(T) < 0)[..., np.newaxis, np.newaxis]
        T_mirror = T - 2 * tf.einsum('Ni,Nk->Nik', U[..., -1], V[..., -1])
        T = tf.where(has_reflection, T_mirror, T)

        if allow_scaling:
            output_scale_mirror = output_scale - 2 * relative_scale * s[:, -1:]
            output_scale = tf.where(has_reflection, output_scale_mirror, output_scale)

    return ((Y - meanY) @ T) * output_scale + meanX


def reconstruct_absolute(coords2d, coords3d_rel, intrinsics):
    inv_intrinsics = tf.linalg.inv(tf.cast(intrinsics, coords2d.dtype))
    coords2d_normalized = tf.matmul(
        to_homogeneous(coords2d), inv_intrinsics, transpose_b=True)[..., :2]
    reconstruct_ref_fn = (
        reconstruct_ref_weakpersp if FLAGS.weak_perspective else reconstruct_ref_fullpersp)
    is_predicted_to_be_in_fov = is_within_fov(coords2d)

    ref = reconstruct_ref_fn(coords2d_normalized, coords3d_rel, is_predicted_to_be_in_fov)
    coords_abs_3d_based = coords3d_rel + tf.expand_dims(ref, 1)
    reference_depth = ref[:, 2]
    relative_depths = coords3d_rel[..., 2]

    coords_abs_2d_based = back_project(coords2d_normalized, relative_depths, reference_depth)
    return tf.where(
        is_predicted_to_be_in_fov[..., tf.newaxis], coords_abs_2d_based, coords_abs_3d_based)


def reconstruct_ref_weakpersp(normalized_2d, coords3d_rel, validity_mask):
    mean3d, stdev3d = tfu.mean_stdev_masked(
        coords3d_rel[..., :2], validity_mask, items_axis=1, dimensions_axis=2)

    mean2d, stdev2d = tfu.mean_stdev_masked(
        normalized_2d[..., :2], validity_mask, items_axis=1, dimensions_axis=2)

    stdev2d = tf.maximum(stdev2d, 1e-5)
    stdev3d = tf.maximum(stdev3d, 1e-5)

    old_mean = tfu.reduce_mean_masked(coords3d_rel, validity_mask, axis=1, keepdims=True)
    new_mean_z = tf.math.divide_no_nan(stdev3d, stdev2d)
    new_mean = to_homogeneous(mean2d) * new_mean_z
    return tf.squeeze(new_mean - old_mean, 1)


def to_homogeneous(x):
    return tf.concat([x, tf.ones_like(x[..., :1])], axis=-1)


def reconstruct_ref_fullpersp(normalized_2d, coords3d_rel, validity_mask):
    """Reconstructs the reference point location.

    Args:
      normalized_2d: normalized image coordinates of the joints
         (without intrinsics applied), shape [batch_size, n_points, 2]
      coords3d_rel: 3D camera coordinate offsets relative to the unknown reference
         point which we want to reconstruct, shape [batch_size, n_points, 3]
      validity_mask: boolean mask of shape [batch_size, n_points] containing True
         where the point is reliable and should be used in the reconstruction

    Returns:
      The 3D reference point in camera coordinates, shape [batch_size, 3]
    """

    def rms_normalize(x):
        scale = tf.sqrt(tf.reduce_mean(tf.square(x)))
        normalized = x / scale
        return scale, normalized

    n_batch = tf.shape(normalized_2d)[0]
    n_points = normalized_2d.shape.as_list()[1]
    eyes = tf.tile(tf.expand_dims(tf.eye(2, 2), 0), [n_batch, n_points, 1])
    scale2d, reshaped2d = rms_normalize(tf.reshape(normalized_2d, [-1, n_points * 2, 1]))
    A = tf.concat([eyes, -reshaped2d], axis=2)

    rel_backproj = normalized_2d * coords3d_rel[:, :, 2:] - coords3d_rel[:, :, :2]
    scale_rel_backproj, b = rms_normalize(tf.reshape(rel_backproj, [-1, n_points * 2, 1]))

    weights = tf.cast(validity_mask, tf.float32) + np.float32(1e-4)
    weights = einops.repeat(weights, 'b j -> b (j c) 1', c=2)

    ref = tf.linalg.lstsq(A * weights, b * weights, l2_regularizer=1e-2, fast=True)
    ref = tf.concat([ref[:, :2], ref[:, 2:] / scale2d], axis=1) * scale_rel_backproj
    return tf.squeeze(ref, axis=-1)


def back_project(camcoords2d, delta_z, z_offset):
    return to_homogeneous(camcoords2d) * tf.expand_dims(delta_z + tf.expand_dims(z_offset, -1), -1)


def is_within_fov(imcoords):
    stride_train = FLAGS.stride_train / FLAGS.final_transposed_conv
    offset = -stride_train / 2 if not FLAGS.centered_stride else 0
    lower = tf.cast(stride_train * 0.75 + offset, tf.float32)
    upper = tf.cast(FLAGS.proc_side - stride_train * 0.75 + offset, tf.float32)
    return tf.reduce_all(tf.logical_and(imcoords >= lower, imcoords <= upper), axis=-1)


def reconstruct_absolute_by_bone_lengths(
        coords25d, intrinsics, bone_lengths_ideal, bones, only_in_fov=True, max_iter=10):
    inv_intrinsics = tf.linalg.inv(tf.cast(intrinsics, coords25d.dtype))
    coords2d_normalized = tf.matmul(
        to_homogeneous(coords25d[..., :2]), inv_intrinsics, transpose_b=True)[..., :2]
    z_relative = center_relative_pose(coords25d[..., 2], None, FLAGS.mean_relative)

    if only_in_fov:
        is_joint_predicted_to_be_in_fov = is_within_fov(coords25d[..., :2])
        bone_weights = tf.cast(tf.stack([
            tf.logical_and(
                is_joint_predicted_to_be_in_fov[:, i],
                is_joint_predicted_to_be_in_fov[:, j])
            for i, j in bones], axis=1), tf.float32) + 1e-8
    else:
        bone_weights = tf.ones([tf.shape(coords25d)[0], len(bones)], dtype=tf.float32)

    maxi = tf.reduce_max(coords2d_normalized, axis=1)
    mini = tf.reduce_min(coords2d_normalized, axis=1)
    projected_size = tf.reduce_max(maxi - mini, axis=-1)
    distance_guess = 1500 / projected_size
    z_ref = optimize_z_offset_by_bones(
        coords2d_normalized, z_relative, bone_lengths_ideal, bones, bone_weights, distance_guess,
        max_iter)
    return back_project(coords2d_normalized, z_relative, z_ref)


def optimize_z_offset_by_bones(
        coords2d_normalized, delta_z, bone_lengths_ideal, bones, bone_weights, initial_guess,
        max_iter=10):
    x = to_homogeneous(coords2d_normalized)
    a = tf.stack([x[:, i] - x[:, j] for i, j in bones], axis=1)
    y = x * tf.expand_dims(delta_z, -1)
    b = tf.stack([y[:, i] - y[:, j] for i, j in bones], axis=1)
    c = tf.reduce_sum(tf.square(a), axis=2)
    d = 2 * tf.reduce_sum(a * b, axis=2)
    e = tf.reduce_sum(tf.square(b), axis=2)

    def residual_fn(z):
        bone_lengths_reprojected = tf.sqrt(tf.square(z) * c + z * d + e)
        return (bone_lengths_reprojected - bone_lengths_ideal) * bone_weights

    initial_guess = tf.cast(initial_guess, tf.float32)[..., np.newaxis]
    initial_guess = tf.broadcast_to(initial_guess, [tf.shape(x)[0], 1])
    solution = levenberg_marquardt(residual_fn, initial_guess, max_iter)[1][0][:, 0]
    return solution
