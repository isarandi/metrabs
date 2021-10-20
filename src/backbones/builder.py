import functools

import numpy as np
import tensorflow as tf
from keras.layers import Lambda
from keras.models import Sequential

import backbones.efficientnet.effnetv2_model as effnetv2_model
import backbones.efficientnet.effnetv2_utils as effnetv2_utils
import backbones.mobilenet_v3
import backbones.resnet
import keras.layers
import tfu
from layers.custom_batchnorms import GhostBatchNormalization
from options import FLAGS


def build_backbone():
    if FLAGS.backbone.startswith('efficientnetv2'):
        bn = effnetv2_utils.BatchNormalization
    else:
        bn = keras.layers.BatchNormalization

    if FLAGS.ghost_bn:
        split = list(map(int, FLAGS.ghost_bn.split(',')))
        bn = functools.partial(GhostBatchNormalization, split=split)

    if FLAGS.backbone.startswith('efficientnetv2'):
        effnetv2_utils.set_batchnorm(bn)
        backbone = effnetv2_model.get_model(FLAGS.backbone, include_top=False)
        preproc_fn = tf_preproc
    elif FLAGS.backbone.startswith('resnet'):
        class MyLayers(keras.layers.VersionAwareLayers):
            def __getattr__(self, name):
                if name == 'BatchNormalization':
                    return bn
                return super().__getattr__(name)

        classname = f'ResNet{FLAGS.backbone[len("resnet"):]}'.replace('-', '_')
        backbone = getattr(backbones.resnet, classname)(
            include_top=False, weights='imagenet',
            input_shape=(None, None, 3), layers=MyLayers())
        if 'V2' in FLAGS.backbone:
            preproc_fn = tf_preproc
        elif 'V1-5' in FLAGS.backbone or 'V1_5' in FLAGS.backbone:
            preproc_fn = torch_preproc
        else:
            preproc_fn = caffe_preproc
    elif FLAGS.backbone.startswith('mobilenet'):
        class MyLayers(backbones.mobilenet_v3.VersionAwareLayers):
            def __getattr__(self, name):
                if name == 'BatchNormalization':
                    return bn
                return super().__getattr__(name)

        arch = FLAGS.backbone
        arch = arch[:-4] if arch.endswith('mini') else arch
        classname = f'MobileNet{arch[len("mobilenet"):]}'
        backbone = getattr(backbones.mobilenet_v3, classname)(
            include_top=False, weights='imagenet', minimalistic=FLAGS.backbone.endswith('mini'),
            input_shape=(FLAGS.proc_side, FLAGS.proc_side, 3), layers=MyLayers(),
            centered_stride=FLAGS.centered_stride, pooling=None)
        preproc_fn = mobilenet_preproc
    else:
        raise Exception

    return Sequential([Lambda(preproc_fn, output_shape=lambda x: x), backbone])


def torch_preproc(x):
    mean_rgb = tf.convert_to_tensor(np.array([0.485, 0.456, 0.406]), tfu.get_dtype())
    stdev_rgb = tf.convert_to_tensor(np.array([0.229, 0.224, 0.225]), tfu.get_dtype())
    normalized = (x - mean_rgb) / stdev_rgb
    return normalized


def caffe_preproc(x):
    mean_rgb = tf.convert_to_tensor(np.array([103.939, 116.779, 123.68]), tfu.get_dtype())
    return tf.cast(255, tfu.get_dtype()) * x - mean_rgb


def tf_preproc(x):
    x = tf.cast(2, tfu.get_dtype()) * x - tf.cast(1, tfu.get_dtype())
    return x


def mobilenet_preproc(x):
    return tf.cast(255, tfu.get_dtype()) * x
