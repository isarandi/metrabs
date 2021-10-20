import tensorflow as tf

import data.datasets3d
import data.joint_filtering
import tfu
from options import FLAGS


def heatmap_to_image(coords, is_training):
    stride = FLAGS.stride_train if is_training else FLAGS.stride_test

    stride //= FLAGS.final_transposed_conv
    last_image_pixel = FLAGS.proc_side - 1
    last_receptive_center = last_image_pixel - (last_image_pixel % stride)
    coords_out = coords * last_receptive_center

    if FLAGS.centered_stride:
        coords_out = coords_out + stride // 2

    return coords_out


def heatmap_to_25d(coords, is_training):
    coords2d = heatmap_to_image(coords[..., :2], is_training)
    return tf.concat([coords2d, coords[..., 2:] * FLAGS.box_size_mm], axis=-1)


def heatmap_to_metric(coords, is_training):
    coords2d = heatmap_to_image(
        coords[..., :2], is_training) * FLAGS.box_size_mm / FLAGS.proc_side
    return tf.concat([coords2d, coords[..., 2:] * FLAGS.box_size_mm], axis=-1)


def matmul_joint_coords(transformation_matrices, coords):
    return tf.einsum('Bij,BCj->BCi', transformation_matrices, coords)


def to_homogeneous(x):
    return tf.concat([x, tf.ones_like(x[..., :1])], axis=-1)


def adjust_skeleton_3dhp_to_mpii(coords3d_pred, joint_info):
    """Move the hips and pelvis towards the neck by a fifth of the pelvis->neck vector
    And move the shoulders up away from the pelvis by 10% of the pelvis->neck vector"""

    j3d = joint_info.ids
    n_joints = joint_info.n_joints
    factor = FLAGS.tdhp_to_mpii_shift_factor
    inverse_factor = -factor / (1 + factor)

    pelvis_neck_vector = coords3d_pred[:, j3d.neck] - coords3d_pred[:, j3d.pelv]
    offset_vector = inverse_factor * pelvis_neck_vector
    n_batch = tfu.dynamic_batch_size(coords3d_pred)

    offsets = []
    for j in range(n_joints):
        if j in (j3d.lhip, j3d.rhip, j3d.pelv):
            offsets.append(offset_vector)
        else:
            offsets.append(tf.zeros([n_batch, 3], dtype=tf.float32))

    offsets = tf.stack([
        tf.zeros([n_batch, 3], dtype=tf.float32)
        if j not in (j3d.lhip, j3d.rhip, j3d.pelv) else offset_vector
        for j in range(n_joints)], axis=1)
    return coords3d_pred + offsets


def align_2d_skeletons(coords_pred, coords_true, joint_validity_mask):
    mean_pred, stdev_pred = tfu.mean_stdev_masked(
        coords_pred, joint_validity_mask, items_axis=1, dimensions_axis=2)
    mean_true, stdev_true = tfu.mean_stdev_masked(
        coords_true, joint_validity_mask, items_axis=1, dimensions_axis=2)
    return tf.math.divide_no_nan(
        coords_pred - mean_pred, stdev_pred) * stdev_true + mean_true


# def align_2d_skeletons(coords_true, coords_pred, joint_validity_mask):
#     mean_pred, stdev_pred = tfu.mean_stdev_masked(
#         coords_pred, joint_validity_mask, items_axis=1, dimensions_axis=2)
#     mean_true, stdev_true = tfu.mean_stdev_masked(
#         coords_true, joint_validity_mask, items_axis=1, dimensions_axis=2)
#
#     coords_pred_result = tf.math.divide_no_nan(coords_pred - mean_pred, stdev_pred) * stdev_true
#     coords_true_result = coords_true - mean_true
#     return coords_true_result, coords_pred_result


def to_orig_cam(x, rot_to_orig_cam, joint_info):
    x = matmul_joint_coords(rot_to_orig_cam, x)
    is_mirrored = tf.linalg.det(rot_to_orig_cam) < 0
    is_mirrored = tfu.expand_dims(is_mirrored, [-1, -1])
    return tf.where(is_mirrored, tf.gather(x, joint_info.mirror_mapping, axis=1), x)


def select_skeleton(coords_src, joint_info_src, skeleton_type_dst):
    if skeleton_type_dst == '':
        return coords_src

    def get_index(name):
        if name + '_' + skeleton_type_dst in joint_info_src.names:
            return joint_info_src.names.index(name + '_h36m')
        else:
            return joint_info_src.names.index(name)

    joint_info_dst = data.datasets3d.get_joint_info(skeleton_type_dst)
    selected_indices = [get_index(name) for name in joint_info_dst.names]
    return tf.gather(coords_src, selected_indices, axis=-2)
