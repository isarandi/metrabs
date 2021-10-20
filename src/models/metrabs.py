import einops
import keras
import keras.layers
import keras.metrics
import numpy as np
import tensorflow as tf
from attrdict import AttrDict

import models.eval_metrics
import models.model_trainer
import models.util
import tfu
import tfu3d
from options import FLAGS


class Metrabs(keras.Model):
    def __init__(self, backbone, joint_info):
        super().__init__()
        self.backbone = backbone
        self.joint_names = tf.Variable(np.array(joint_info.names), trainable=False)
        self.joint_edges = tf.Variable(np.array(joint_info.stick_figure_edges), trainable=False)
        self.joint_info = joint_info
        n_raw_points = 32 if FLAGS.transform_coords else joint_info.n_joints
        self.heatmap_heads = MetrabsHeads(n_points=n_raw_points)
        if FLAGS.transform_coords:
            self.recombination_weights = tf.constant(np.load('32_to_122'))

    def call(self, inp, training=None):
        image, intrinsics = inp
        features = self.backbone(image, training=training)
        coords2d, coords3d = self.heatmap_heads(features, training=training)
        coords3d_abs = tfu3d.reconstruct_absolute(coords2d, coords3d, intrinsics)
        if FLAGS.transform_coords:
            coords3d_abs = self.latent_points_to_joints(coords3d_abs)

        return coords3d_abs

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float16),
        tf.TensorSpec(shape=(None, 3, 3), dtype=tf.float32)])
    def predict_multi(self, image, intrinsic_matrix):
        # This function is needed to avoid having to go through Keras' __call__
        # in the exported SavedModel, which causes all kinds of problems.
        return self.call((image, intrinsic_matrix), training=False)

    def latent_points_to_joints(self, points):
        return tfu3d.linear_combine_points(points, self.recombination_weights)


class MetrabsHeads(keras.layers.Layer):
    def __init__(self, n_points):
        super().__init__()
        self.n_points = n_points
        self.n_outs = [self.n_points, FLAGS.depth * self.n_points]
        self.conv_final = keras.layers.Conv2D(filters=sum(self.n_outs), kernel_size=1)

    def call(self, inp, training=None):
        x = self.conv_final(inp)
        logits2d, logits3d = tf.split(x, self.n_outs, axis=tfu.channel_axis())
        current_format = 'b h w (d j)' if tfu.get_data_format() == 'NHWC' else 'b (d j) h w'
        logits3d = einops.rearrange(logits3d, f'{current_format} -> b h w d j', j=self.n_points)
        coords3d = tfu.soft_argmax(tf.cast(logits3d, tf.float32), axis=[2, 1, 3])
        coords3d_rel_pred = models.util.heatmap_to_metric(coords3d, training)
        coords2d = tfu.soft_argmax(tf.cast(logits2d, tf.float32), axis=tfu.image_axes()[::-1])
        coords2d_pred = models.util.heatmap_to_image(coords2d, training)
        return coords2d_pred, coords3d_rel_pred


class MetrabsTrainer(models.model_trainer.ModelTrainer):
    def __init__(self, metrabs_model, joint_info, joint_info2d=None, global_step=None):
        super().__init__(global_step)
        self.global_step = global_step
        self.joint_info = joint_info
        self.joint_info_2d = joint_info2d
        self.model = metrabs_model
        inp = keras.Input(shape=(None, None, 3), dtype=tfu.get_dtype())
        intr = keras.Input(shape=(3, 3), dtype=tf.float32)
        self.model((inp, intr), training=False)

    def forward_train(self, inps, training):
        preds = AttrDict()

        image_both = tf.concat([inps.image, inps.image_2d], axis=0)
        features = self.model.backbone(image_both, training=training)
        coords2d_pred_both, coords3d_rel_pred_both = self.model.heatmap_heads(
            features, training=training)
        batch_sizes = [t.shape.as_list()[0] for t in [inps.image, inps.image_2d]]
        preds.coords2d_pred, preds.coords2d_pred_2d = tf.split(
            coords2d_pred_both, batch_sizes, axis=0)
        preds.coords3d_rel_pred, preds.coords3d_rel_pred_2d = tf.split(
            coords3d_rel_pred_both, batch_sizes, axis=0)

        if FLAGS.transform_coords:
            l2j = self.model.latent_points_to_joints
            preds.coords2d_pred_2d = l2j(preds.coords2d_pred_2d)
            preds.coords3d_rel_pred_2d = l2j(preds.coords3d_rel_pred_2d)
            preds.coords2d_pred_latent = preds.coords2d_pred
            preds.coords2d_pred = l2j(preds.coords2d_pred_latent)
            preds.coords3d_rel_pred_latent = preds.coords3d_rel_pred
            preds.coords3d_rel_pred = l2j(preds.coords3d_rel_pred_latent)
            preds.coords3d_pred_abs = l2j(tfu3d.reconstruct_absolute(
                preds.coords2d_pred_latent, preds.coords3d_rel_pred_latent, inps.intrinsics))
        else:
            preds.coords3d_pred_abs = tfu3d.reconstruct_absolute(
                preds.coords2d_pred, preds.coords3d_rel_pred, inps.intrinsics)

        joint_ids_3d = [
            [self.joint_info.ids[n2] for n2 in self.joint_info.names if n2.startswith(n1)]
            for n1 in self.joint_info_2d.names]

        def get_2dlike_joints(coords):
            return tf.stack(
                [tf.reduce_mean(tf.gather(coords, ids, axis=1)[..., :2], axis=1)
                 for ids in joint_ids_3d], axis=1)

        # numbers mean: 3d head, 2d dataset joints, 2d batch
        preds.coords32d_pred_2d = get_2dlike_joints(preds.coords3d_rel_pred_2d)
        preds.coords22d_pred_2d = get_2dlike_joints(preds.coords2d_pred_2d)
        return preds

    def compute_losses(self, inps, preds):
        losses = AttrDict()

        if FLAGS.scale_agnostic_loss:
            mean_true, scale_true = tfu.mean_stdev_masked(
                inps.coords3d_true, inps.joint_validity_mask, items_axis=1, dimensions_axis=2)
            mean_pred, scale_pred = tfu.mean_stdev_masked(
                preds.coords3d_rel_pred, inps.joint_validity_mask, items_axis=1, dimensions_axis=2)
            coords3d_pred_rootrel = tf.math.divide_no_nan(
                preds.coords3d_rel_pred - mean_pred, scale_pred) * scale_true
            coords3d_true_rootrel = inps.coords3d_true - mean_true
        else:
            coords3d_true_rootrel = tfu3d.center_relative_pose(
                inps.coords3d_true, inps.joint_validity_mask, FLAGS.mean_relative)
            coords3d_pred_rootrel = tfu3d.center_relative_pose(
                preds.coords3d_rel_pred, inps.joint_validity_mask, FLAGS.mean_relative)

        rootrel_absdiff = tf.abs((coords3d_true_rootrel - coords3d_pred_rootrel) / 1000)
        losses.loss3d = tfu.reduce_mean_masked(rootrel_absdiff, inps.joint_validity_mask)

        if FLAGS.scale_agnostic_loss:
            _, scale_true = tfu.mean_stdev_masked(
                inps.coords3d_true, inps.joint_validity_mask, items_axis=1, dimensions_axis=2,
                fixed_ref=tf.zeros_like(inps.coords3d_true))
            _, scale_pred = tfu.mean_stdev_masked(
                preds.coords3d_pred_abs, inps.joint_validity_mask, items_axis=1, dimensions_axis=2,
                fixed_ref=tf.zeros_like(inps.coords3d_true))
            preds.coords3d_pred_abs = tf.math.divide_no_nan(
                preds.coords3d_pred_abs, scale_pred) * scale_true

        if self.global_step > 5000:
            absdiff = tf.abs((inps.coords3d_true - preds.coords3d_pred_abs) / 1000)
            losses.loss3d_abs = tfu.reduce_mean_masked(absdiff, inps.joint_validity_mask)
        else:
            losses.loss3d_abs = tf.constant(0, tf.float32)

        scale_2d = 1 / FLAGS.proc_side * FLAGS.box_size_mm / 1000
        losses.loss23d = tfu.reduce_mean_masked(
            tf.abs((inps.coords2d_true - preds.coords2d_pred) * scale_2d),
            inps.joint_validity_mask)

        preds.coords32d_pred_2d = models.util.align_2d_skeletons(
            preds.coords32d_pred_2d, inps.coords2d_true_2d, inps.joint_validity_mask_2d)
        losses.loss32d = tfu.reduce_mean_masked(
            tf.abs((inps.coords2d_true_2d - preds.coords32d_pred_2d) * scale_2d),
            inps.joint_validity_mask_2d)
        losses.loss22d = tfu.reduce_mean_masked(
            tf.abs((inps.coords2d_true_2d - preds.coords22d_pred_2d) * scale_2d),
            inps.joint_validity_mask_2d)

        losses3d = [losses.loss3d, losses.loss23d, FLAGS.absloss_factor * losses.loss3d_abs]
        losses2d = [losses.loss22d, losses.loss32d]
        losses.loss = tf.add_n(losses3d) + FLAGS.loss2d_factor * tf.add_n(losses2d)
        return losses

    def compute_metrics(self, inps, preds):
        return models.eval_metrics.compute_pose3d_metrics(inps, preds)

    def forward_test(self, inps):
        preds = AttrDict()
        features = self.model.backbone(inps.image, training=False)
        preds.coords2d_pred, preds.coords3d_rel_pred = self.model.heatmap_heads(
            features, training=False)
        preds.coords3d_pred_abs = tfu3d.reconstruct_absolute(
            preds.coords2d_pred, preds.coords3d_rel_pred, inps.intrinsics)

        if FLAGS.transform_coords:
            l2j = self.model.latent_points_to_joints
            preds.coords2d_pred = l2j(preds.coords2d_pred)
            preds.coords3d_rel_pred = l2j(preds.coords3d_rel_pred)
            preds.coords3d_pred_abs = l2j(preds.coords3d_pred_abs)

        return preds
