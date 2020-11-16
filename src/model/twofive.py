import tensorflow as tf

import data.datasets3d
import data.datasets2d
import model.architectures
import model.util
import tfu
import tfu3d
from options import FLAGS
from model.bone_length_based_backproj import get_bone_lengths, optimize_z_offset_by_bones, \
    optimize_z_offset_by_bones_tensor


def build_25d_model(joint_info, t):
    if not tfu.is_training():
        return build_25d_inference_model(joint_info, t)

    if FLAGS.batchnorm_together_2d3d:
        batch_size3d = tfu.dynamic_batch_size(t.x)
        batch_size2d = tfu.dynamic_batch_size(t.x_2d)

        # Concatenate the 3D and the 2D batch
        x_both_batches = tf.concat([t.x, t.x_2d], axis=0)
        coords25d_pred_both = predict_25d(x_both_batches, joint_info)
        # Split the results (3D batch and 2D batch)
        t.coords25d_pred, t.coords25d_pred_2d = tf.split(
            coords25d_pred_both, [batch_size3d, batch_size2d])
    else:
        # Send the 2D and the 3D batch separately through the network,
        # so each gets normalized only within itself
        t.coords25d_pred = predict_25d(t.x, joint_info)
        t.coords25d_pred_2d = predict_25d(t.x_2d, joint_info)

    if FLAGS.dataset == 'mpi_inf_3dhp':
        t.coords25d_pred_2d = model.util.adjust_skeleton_3dhp_to_mpii(
            t.coords25d_pred_2d, joint_info)

    joint_info_2d = data.datasets2d.get_dataset(FLAGS.dataset2d).joint_info
    joint_ids_3d = [joint_info.ids[name] for name in joint_info_2d.names]
    t.coords2d_pred2d = tf.gather(t.coords25d_pred_2d[..., :2], joint_ids_3d, axis=1)
    t.coords2d_pred = t.coords25d_pred[..., :2]

    # LOSS 3D BATCH
    scale2d = 1 / FLAGS.proc_side * FLAGS.box_size_mm / 1000
    t.loss2d_3d = tf.reduce_mean(tf.abs(t.coords2d_true - t.coords2d_pred)) * scale2d

    z_ref = t.coords3d_true[..., 2] - t.coords3d_true[:, -1:, 2] + 0.5 * FLAGS.box_size_mm
    t.loss_z = tf.reduce_mean(tf.abs(z_ref - t.coords25d_pred[..., 2])) / 1000

    # LOSS 2D BATCH
    t.loss2d = tfu.reduce_mean_masked(
        tf.abs(t.coords2d_true2d - t.coords2d_pred2d), t.joint_validity_mask2d) * scale2d

    t.loss3d = (t.loss2d_3d * 2 + t.loss_z) / 3
    t.loss = t.loss3d + FLAGS.loss2d_factor * t.loss2d

    # POST-PROCESSING
    if FLAGS.bone_length_dataset:
        dataset = data.datasets3d.get_dataset(FLAGS.bone_length_dataset)
    else:
        dataset = data.datasets3d.get_dataset(FLAGS.dataset)

    delta_z_pred = t.coords25d_pred[..., 2] - t.coords25d_pred[:, -1:, 2]
    if FLAGS.train_on == 'trainval':
        target_bone_lengths = dataset.trainval_bones
    else:
        target_bone_lengths = dataset.train_bones

    camcoords2d_homog = model.util.matmul_joint_coords(
        t.inv_intrinsics, model.util.to_homogeneous(t.coords2d_pred))
    z_offset = optimize_z_offset_by_bones(
        camcoords2d_homog, delta_z_pred, target_bone_lengths, joint_info.stick_figure_edges)
    t.coords3d_pred = model.util.back_project(camcoords2d_homog, delta_z_pred, z_offset)


def predict_25d(im, joint_info):
    stride = FLAGS.stride_train if tfu.is_training() else FLAGS.stride_test
    net_output = model.architectures.resnet(
        im,  n_outs=[FLAGS.depth * joint_info.n_joints], scope='MainPart',
        reuse=tf.compat.v1.AUTO_REUSE, stride=stride, centered_stride=FLAGS.centered_stride,
        resnet_name=FLAGS.architecture)[0]

    logits = tfu.std_to_nchw(net_output)
    side = tfu.static_image_shape(net_output)[0]
    logits = tf.reshape(logits, [-1, FLAGS.depth, joint_info.n_joints, side, side])
    logits = tf.transpose(logits, [0, 2, 3, 4, 1])
    coords = tfu.soft_argmax(logits, axis=[3, 2, 4])
    return model.util.heatmap_to_25d(coords)


def build_25d_inference_model(joint_info, t):
    t.x = tf.identity(t.x, 'input')
    coords25d = predict_25d(t.x, joint_info)
    t.coords2d_pred = coords25d[..., :2]
    camcoords2d_homog = model.util.matmul_joint_coords(
        t.inv_intrinsics, model.util.to_homogeneous(t.coords2d_pred))
    delta_z_pred = (coords25d[..., 2] - coords25d[:, -1:, 2])

    if 'bone-lengths' in FLAGS.scale_recovery:
        if FLAGS.bone_length_dataset:
            dataset = data.datasets3d.get_dataset(FLAGS.bone_length_dataset)
        else:
            dataset = data.datasets3d.get_dataset(FLAGS.dataset)

        if FLAGS.scale_recovery == 'bone-lengths-true':
            bone_lengths_true = get_bone_lengths(t.coords3d_true, joint_info)
            z_offset = optimize_z_offset_by_bones_tensor(
                camcoords2d_homog, delta_z_pred, bone_lengths_true,
                joint_info.stick_figure_edges)
        else:
            target_bone_lengths = (
                dataset.trainval_bones if FLAGS.train_on == 'trainval' else dataset.train_bones)
            z_offset = optimize_z_offset_by_bones(
                camcoords2d_homog, delta_z_pred, target_bone_lengths,
                joint_info.stick_figure_edges)
        t.coords3d_pred = model.util.back_project(camcoords2d_homog, delta_z_pred, z_offset)
    elif FLAGS.scale_recovery == 'true-root-depth':
        t.coords3d_pred = model.util.back_project(
            camcoords2d_homog, delta_z_pred, t.coords3d_true[:, -1, 2])

    t.coords3d_pred_rootrel = tf.identity(tfu3d.root_relative(t.coords3d_pred), 'pred')
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
