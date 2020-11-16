import tensorflow as tf

import data.datasets2d
import data.datasets3d
import model.architectures
import model.util
import tfu
import tfu3d
from options import FLAGS


def build_metro_model(joint_info, t):
    if not tfu.is_training():
        return build_metro_inference_model(joint_info, t)

    # Generate predictions for both the 3D and the 2D batch
    if FLAGS.batchnorm_together_2d3d:
        batch_size3d = tfu.dynamic_batch_size(t.x)
        batch_size2d = tfu.dynamic_batch_size(t.x_2d)

        # Concatenate the 3D and the 2D batch
        x_both_batches = tf.concat([t.x, t.x_2d], axis=0)
        coords3d_pred_both = predict_metro(x_both_batches, joint_info)
        # Split the results (3D batch and 2D batch)
        t.coords3d_pred, t.coords3d_pred2d = tf.split(
            coords3d_pred_both, [batch_size3d, batch_size2d])
    else:
        # Send the 2D and the 3D batch separately through the network,
        # so each gets normalized only within itself
        t.coords3d_pred = predict_metro(t.x, joint_info)
        t.coords3d_pred2d = predict_metro(t.x_2d, joint_info)

    # Loss for 3D batch
    t.coords3d_true_rootrel = tfu3d.center_relative_pose(
        t.coords3d_true, t.joint_validity_mask, FLAGS.mean_relative)
    t.coords3d_pred_rootrel = tfu3d.center_relative_pose(
        t.coords3d_pred, t.joint_validity_mask, FLAGS.mean_relative)

    rootrel_absdiff = tf.abs(t.coords3d_true_rootrel - t.coords3d_pred_rootrel)
    t.loss3d = tfu.reduce_mean_masked(rootrel_absdiff, t.joint_validity_mask) / 1000

    ## Loss for 2D batch
    # Pick out the joints that correspond to the ones labeled in the 2D dataset
    joint_info_2d = data.datasets2d.get_dataset(FLAGS.dataset2d).joint_info
    joint_ids_3d = [joint_info.ids[name] for name in joint_info_2d.names]
    t.coords32d_pred2d = tf.gather(t.coords3d_pred2d, joint_ids_3d, axis=1)[..., :2]

    scale_factor2d = 1 / FLAGS.proc_side * FLAGS.box_size_mm / 1000
    t.coords32d_pred2d = model.util.align_2d_skeletons(
        t.coords2d_true2d, t.coords32d_pred2d, t.joint_validity_mask2d)
    t.loss2d = tfu.reduce_mean_masked(
        tf.abs(t.coords2d_true2d - t.coords32d_pred2d), t.joint_validity_mask2d) * scale_factor2d

    t.loss = t.loss3d + FLAGS.loss2d_factor * t.loss2d


def build_metro_inference_model(joint_info, t):
    t.coords3d_pred = predict_metro(t.x, joint_info)
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


def predict_metro(im, joint_info):
    stride = FLAGS.stride_train if tfu.is_training() else FLAGS.stride_test
    # 1. Feed image through backbone
    logits = model.architectures.resnet(
        im, n_outs=[FLAGS.depth * joint_info.n_joints], scope='MainPart',
        reuse=tf.compat.v1.AUTO_REUSE, stride=stride, centered_stride=FLAGS.centered_stride,
        resnet_name=FLAGS.architecture)[0]
    logits = tfu.std_to_nchw(logits)
    side = tfu.static_shape(logits)[2]

    # 2. Reshape the 3D heatmap logits to actually be 3D: [batch, joints, H, W, D]
    logits = tf.reshape(logits, [-1, FLAGS.depth, joint_info.n_joints, side, side])
    logits = tf.transpose(logits, [0, 2, 3, 4, 1])

    # 3. Decode the heatmap coordinates using soft-argmax, resulting in values between 0 and 1
    coords3d_raw = tfu.soft_argmax(logits, [3, 2, 4])

    # 4. Scale and shift the normalized heatmap coordinates to get metric and pixel values
    coords3d_pred = model.util.heatmap_to_metric(coords3d_raw)
    return coords3d_pred


def predict_direct(im, joint_info):
    stride = FLAGS.stride_train if tfu.is_training() else FLAGS.stride_test
    values = model.architectures.resnet(
        im, n_outs=[3 * joint_info.n_joints], scope='MainPart',
        reuse=tf.compat.v1.AUTO_REUSE, stride=stride, centered_stride=FLAGS.centered_stride,
        resnet_name=FLAGS.architecture, global_pool=True)[0]
    return tf.reshape(values, [-1, joint_info.n_joints, 3]) * 1000
