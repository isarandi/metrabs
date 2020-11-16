import numpy as np
import tensorflow as tf

import tfu
import util3d


def pck(pred, labels, joint_validity_mask, reference_size, max_relative_distance):
    rel_dist = relative_distance(pred, labels, reference_size)
    is_correct = tf.cast(rel_dist <= max_relative_distance, tf.float32)
    return tfu.reduce_mean_masked(is_correct, joint_validity_mask, axis=0)


def auc(pred, labels, joint_validity_mask, reference_size, max_relative_distance):
    rel_dist = relative_distance(pred, labels, reference_size)
    score = tf.maximum(rel_dist, max_relative_distance)
    return tfu.reduce_mean_masked(score, joint_validity_mask, axis=0)


def relative_distance(pred, labels, reference_size):
    return tf.norm(pred - labels, axis=2) / tf.expand_dims(reference_size, axis=1)


def root_relative(coords3d):
    center = coords3d[:, -1:, :]
    return coords3d - center


def rigid_align(coords_pred, coords_true, *, joint_validity_mask=None, scale_align=False):
    def func(_coords_pred, _coords_true, _joint_validity_mask):
        return util3d.rigid_align_many(
            _coords_pred, _coords_true,
            joint_validity_mask=_joint_validity_mask, scale_align=scale_align)

    if joint_validity_mask is None:
        joint_validity_mask = tf.ones_like(coords_pred[..., 0], dtype=tf.bool)

    return tfu.py_func_with_shapes(
        func=func, inp=[coords_pred, coords_true, joint_validity_mask],
        output_types=tf.float32, output_shapes=tfu.static_shape(coords_pred))


def center_relative_pose(coords3d, joint_validity_mask=None, center_is_mean=False):
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
                center = tfu.reduce_mean_masked(coords3d, joint_validity_mask, axis=1,
                                                keepdims=True)
    else:
        center = coords3d[:, -1:, :]
    return coords3d - center


@tfu.in_variable_scope('TransformSkeleton')
def transform_coords(coords, n_output_joints):
    """Learned linear combination of joints positions, either for 2D or 3D"""

    def normalize_weights(w):
        return w / tf.reduce_sum(w, axis=0, keepdims=True)

    n_input_joints = tfu.static_shape(coords)[1]
    initializer = tf.constant_initializer(np.eye(n_input_joints), dtype=coords.dtype)
    weights = tf.get_variable(
        name='skeleton_weights', shape=(n_input_joints, n_output_joints), dtype=coords.dtype,
        initializer=initializer, trainable=True, regularizer=None,
        constraint=normalize_weights)
    return tf.einsum('bjc,jJ->bJc', coords, normalize_weights(weights))