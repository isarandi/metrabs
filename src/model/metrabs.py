import numpy as np
import tensorflow as tf

import data.datasets3d
import data.datasets2d
import model.architectures
import model.util
import tfu
import tfu3d
from options import FLAGS


def build_metrabs_model(joint_info, t):
    if not tfu.is_training():
        return build_metrabs_inference_model(joint_info, t)

    # Generate predictions for both the 3D and the 2D batch
    if FLAGS.batchnorm_together_2d3d:
        batch_size3d = tfu.dynamic_batch_size(t.x)
        batch_size2d = tfu.dynamic_batch_size(t.x_2d)

        # Concatenate the 3D and the 2D batch
        x_both_batches = tf.concat([t.x, t.x_2d], axis=0)
        coords2d_pred_both, coords3d_rel_pred_both = predict_heads_metrabs(
            x_both_batches, joint_info)
        # Split the results (3D batch and 2D batch)
        t.coords2d_pred, t.coords2d_pred2d = tf.split(
            coords2d_pred_both, [batch_size3d, batch_size2d])
        t.coords3d_rel_pred, t.coords3d_pred2d = tf.split(
            coords3d_rel_pred_both, [batch_size3d, batch_size2d])
    else:
        # Send the 2D and the 3D batch separately through the network,
        # so each gets normalized only within itself
        t.coords2d_pred, t.coords3d_rel_pred = predict_heads_metrabs(t.x, joint_info)
        t.coords2d_pred2d, t.coords3d_pred2d = predict_heads_metrabs(t.x_2d, joint_info)

    # Reconstruct absolute pose only on the 3D batch
    t.coords3d_pred = tf.cond(
        t.global_step > 50,
        lambda: reconstruct_absolute(t.coords2d_pred, t.coords3d_rel_pred, t.inv_intrinsics),
        lambda: t.coords3d_rel_pred)

    ######
    # LOSSES FOR 3D BATCH
    ######
    #
    # Loss on 2D head for 3D batch
    scale_factor2d = 1 / FLAGS.proc_side * FLAGS.box_size_mm / 1000
    t.loss23d = tfu.reduce_mean_masked(
        tf.abs(t.coords2d_true - t.coords2d_pred), t.joint_validity_mask) * scale_factor2d

    # Loss on 3D head (relative) for 3D batch
    t.coords3d_true_rootrel = tfu3d.center_relative_pose(
        t.coords3d_true, t.joint_validity_mask, FLAGS.mean_relative)
    t.coords3d_pred_rootrel = tfu3d.center_relative_pose(
        t.coords3d_rel_pred, t.joint_validity_mask, FLAGS.mean_relative)
    rootrel_absdiff = tf.abs(t.coords3d_true_rootrel - t.coords3d_pred_rootrel)
    t.loss3d = tfu.reduce_mean_masked(rootrel_absdiff, t.joint_validity_mask) / 1000

    # Loss on absolute reconstruction for 3D batch
    absloss_factor = tf.where(
        t.global_step > 5000,
        tf.convert_to_tensor(FLAGS.absloss_factor, dtype=tf.float32),
        tf.convert_to_tensor(0, dtype=tf.float32))
    absdiff = tf.abs(t.coords3d_true - t.coords3d_pred)
    t.loss3d_abs = absloss_factor * tfu.reduce_mean_masked(absdiff, t.joint_validity_mask) / 1000
    losses3d = [t.loss3d, t.loss23d, absloss_factor * t.loss3d_abs]

    ######
    # LOSSES FOR 2D BATCH
    ######
    #
    # Pick out the joints that correspond to the ones labeled in the 2D dataset
    joint_info_2d = data.datasets2d.get_dataset(FLAGS.dataset2d).joint_info
    joint_ids_3d = [joint_info.ids[name] for name in joint_info_2d.names]
    t.coords32d_pred2d = tf.gather(t.coords3d_pred2d, joint_ids_3d, axis=1)[..., :2]
    t.coords22d_pred2d = tf.gather(t.coords2d_pred2d, joint_ids_3d, axis=1)[..., :2]

    # Loss on 2D head for 2D batch
    t.loss22d = tfu.reduce_mean_masked(
        tf.abs(t.coords2d_true2d - t.coords22d_pred2d),
        t.joint_validity_mask2d) * scale_factor2d

    # Loss on 3D head for 2D batch
    t.coords32d_pred2d = model.util.align_2d_skeletons(
        t.coords2d_true2d, t.coords32d_pred2d, t.joint_validity_mask2d)
    t.loss32d = tfu.reduce_mean_masked(
        tf.abs(t.coords2d_true2d - t.coords32d_pred2d),
        t.joint_validity_mask2d) * scale_factor2d
    losses2d = [t.loss22d, t.loss32d]

    t.loss = tf.add_n(losses3d) + FLAGS.loss2d_factor * tf.add_n(losses2d)


def build_metrabs_inference_model(joint_info, t):
    t.coords2d_pred, t.coords3d_rel_pred = predict_heads_metrabs(t.x, joint_info)
    t.coords3d_pred = reconstruct_absolute(t.coords2d_pred, t.coords3d_rel_pred, t.inv_intrinsics)

    if 'rot_to_orig_cam' in t:
        t.coords3d_pred_orig_cam = model.util.to_orig_cam(
            t.coords3d_pred, t.rot_to_orig_cam, joint_info)
    if 'rot_to_world' in t:
        t.coords3d_pred_world = model.util.to_orig_cam(
            t.coords3d_pred, t.rot_to_world, joint_info) + tf.expand_dims(t.cam_loc, 1)

    if 'coords3d_true' in t:
        t.coords3d_true_rootrel = tfu3d.root_relative(t.coords3d_true)
        if 'rot_to_orig_cam' in t:
            t.coords3d_true_orig_cam = model.util.to_orig_cam(
                t.coords3d_true, t.rot_to_orig_cam, joint_info)
        if 'rot_to_world' in t:
            t.coords3d_true_world = model.util.to_orig_cam(
                t.coords3d_true, t.rot_to_world, joint_info) + tf.expand_dims(t.cam_loc, 1)


def predict_heads_metrabs(im, joint_info):
    stride = FLAGS.stride_train if tfu.is_training() else FLAGS.stride_test

    if FLAGS.metrabs_plus:
        n_outs = [FLAGS.depth * joint_info.n_joints, FLAGS.depth * joint_info.n_joints]
    else:
        n_outs = [joint_info.n_joints, FLAGS.depth * joint_info.n_joints]
    # 1. Feed image through backbone
    logits2d, logits3d = model.architectures.resnet(
        im, n_outs=n_outs, scope='MainPart', reuse=tf.compat.v1.AUTO_REUSE, stride=stride,
        centered_stride=FLAGS.centered_stride, resnet_name=FLAGS.architecture)
    logits2d = tfu.std_to_nchw(logits2d)
    logits3d = tfu.std_to_nchw(logits3d)
    side = tfu.static_shape(logits3d)[2]

    # 2. Reshape the 3D heatmap logits to actually be 3D: [batch, joints, H, W, D]
    logits3d = tf.reshape(logits3d, [-1, FLAGS.depth, joint_info.n_joints, side, side])
    logits3d = tf.transpose(logits3d, [0, 2, 3, 4, 1])

    # 3. Decode the heatmap coordinates using soft-argmax, resulting in values between 0 and 1
    coords3d_raw = tfu.soft_argmax(logits3d, [3, 2, 4])

    if FLAGS.metrabs_plus:
        logits2d = tf.reshape(logits2d, [-1, FLAGS.depth, joint_info.n_joints, side, side])
        logits2d = tf.transpose(logits2d, [0, 2, 3, 4, 1])
        coords2d_raw = tfu.soft_argmax(logits2d, [3, 2, 4])
        coords2d_raw = tfu3d.transform_coords(
            coords2d_raw, joint_info.n_joints, reuse=tf.compat.v1.AUTO_REUSE, scope='transform')
        coords3d_raw = tfu3d.transform_coords(
            coords3d_raw, joint_info.n_joints, reuse=tf.compat.v1.AUTO_REUSE, scope='transform')
        coords2d_pred = model.util.heatmap_to_25d(coords2d_raw)
    else:
        coords2d_raw = tfu.soft_argmax(logits2d, [3, 2])
        coords2d_pred = model.util.heatmap_to_image(coords2d_raw)

    # 4. Scale and shift the normalized heatmap coordinates to get metric and pixel values
    coords3d_rel_pred = model.util.heatmap_to_metric(coords3d_raw)
    return coords2d_pred, coords3d_rel_pred


def reconstruct_absolute(coords2d, coords3d_rel, inv_intrinsics):
    coords2d_normalized = model.util.matmul_joint_coords(
        inv_intrinsics, model.util.to_homogeneous(coords2d[:, :, :2]))
    reconstruct_ref = reconstruct_ref_weak if FLAGS.weak_perspective else reconstruct_ref_strong

    is_predicted_to_be_in_fov = tf.reduce_all(
        tf.logical_and(
            coords2d[:, :, :2] >= tf.cast(FLAGS.stride_train, tf.float32),
            coords2d[:, :, :2] <= tf.cast(FLAGS.proc_side - FLAGS.stride_train, tf.float32)),
        axis=-1)

    ref = reconstruct_ref(coords2d_normalized[:, :, :2], coords3d_rel, is_predicted_to_be_in_fov)
    coords_abs_3d_based = coords3d_rel + tf.expand_dims(ref, 1)

    if FLAGS.metrabs_plus:
        reference_depth = ref[:, 2] + tfu.reduce_mean_masked(
            coords3d_rel[:, :, 2] - coords2d[:, :, 2], is_predicted_to_be_in_fov, axis=1)
        relative_depths = coords2d[:, :, 2]
    else:
        reference_depth = ref[:, 2]
        relative_depths = coords3d_rel[:, :, 2]

    coords_abs_2d_based = model.util.back_project(
        coords2d_normalized, relative_depths, reference_depth)
    is_predicted_to_be_in_fov = tf.tile(tf.expand_dims(is_predicted_to_be_in_fov, -1), [1, 1, 3])
    return tf.where(is_predicted_to_be_in_fov, coords_abs_2d_based, coords_abs_3d_based)


def reconstruct_ref_weak(normalized_2d, coords3d_rel, validity_mask):
    mean3d, stdev3d = tfu.mean_stdev_masked(
        coords3d_rel[..., :2], validity_mask, items_axis=1, dimensions_axis=2)

    mean2d, stdev2d = tfu.mean_stdev_masked(
        normalized_2d[..., :2], validity_mask, items_axis=1, dimensions_axis=2)

    stdev2d = tf.maximum(stdev2d, 1e-5)
    stdev3d = tf.maximum(stdev3d, 1e-5)

    old_mean = tfu.reduce_mean_masked(coords3d_rel, validity_mask, axis=1, keepdims=True)
    new_mean_z = tf.math.divide_no_nan(stdev3d, stdev2d)
    new_mean = model.util.to_homogeneous(mean2d) * new_mean_z
    return tf.squeeze(new_mean - old_mean, 1)


def reconstruct_ref_strong(normalized_2d, coords3d_rel, validity_mask):
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

    def root_mean_square(x):
        return tf.sqrt(tf.reduce_mean(tf.square(x)))

    n_batch = tfu.dynamic_batch_size(normalized_2d)
    n_points = tfu.static_shape(normalized_2d)[1]

    eyes = tf.tile(tf.expand_dims(tf.eye(2, 2), 0), [n_batch, n_points, 1])
    reshaped2d = tf.reshape(normalized_2d, [-1, n_points * 2, 1])
    scale2d = root_mean_square(reshaped2d)
    A = tf.concat([eyes, -reshaped2d / scale2d], axis=2)

    rel_backproj = normalized_2d * coords3d_rel[:, :, 2:] - coords3d_rel[:, :, :2]
    scale_rel_backproj = root_mean_square(rel_backproj)
    b = tf.reshape(rel_backproj / scale_rel_backproj, [-1, n_points * 2, 1])

    weights = tf.cast(validity_mask, tf.float32) + np.float32(1e-4)
    weights = tf.reshape(tf.tile(tf.expand_dims(weights, -1), [1, 1, 2]), [-1, n_points * 2, 1])

    ref = tf.linalg.lstsq(A * weights, b * weights, fast=True)
    ref = tf.concat([ref[:, :2], ref[:, 2:] / scale2d], axis=1) * scale_rel_backproj

    return ref
