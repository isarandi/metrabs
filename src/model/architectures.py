import tensorflow as tf
import tf_slim as slim

import model.resnet_v2
import tfu
from options import FLAGS


def resnet_arg_scope():
    batch_norm_params = dict(
        decay=0.997, epsilon=1e-5, scale=True, is_training=tfu.is_training(), fused=True,
        data_format=tfu.data_format())

    with slim.arg_scope(
            [slim.conv2d, slim.conv3d],
            weights_regularizer=slim.l2_regularizer(1e-4),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


@tfu.in_variable_scope('Resnet', mixed_precision=True)
def resnet(
        inp, n_outs, stride=16, centered_stride=False, global_pool=False, resnet_name='resnet_v2_50'):
    with slim.arg_scope(resnet_arg_scope()):
        if FLAGS.compatibility_mode:
            x = tf.cast(inp, tfu.get_dtype())
        else:
            x = tf.cast(inp * 2 - 1, tfu.get_dtype())

        resnet_fn = getattr(model.resnet_v2, resnet_name)
        xs, end_points = resnet_fn(
            x, num_classes=n_outs, is_training=tfu.is_training(), global_pool=global_pool,
            output_stride=stride, centered_stride=centered_stride)
        xs = [tf.cast(x, tf.float32) for x in xs]
        return xs
