import functools

import keras
import tensorflow as tf
from keras.utils import conv_utils
import keras.layers


class Conv2DDenseSame(keras.layers.Conv2D):
    def __init__(
            self, *args, dilation_rate_test=None, strides_test=None, bottomright_stride=False,
            bottomright_stride_test=False, **kwargs):
        super(Conv2DDenseSame, self).__init__(*args, padding='VALID', **kwargs)
        self.si = StridingInfo(
            strides=self.strides, strides_test=strides_test, dilation_rate=self.dilation_rate,
            dilation_rate_test=dilation_rate_test, bottomright_stride=bottomright_stride,
            bottomright_stride_test=bottomright_stride_test)

        self.preproc_train = self._make_preproc_layer(
            self.si.dilation_rate, self.si.bottomright_stride)
        self.preproc_test = self._make_preproc_layer(
            self.si.dilation_rate_test, self.si.bottomright_stride_test)

    def build(self, input_shape):
        self._convolution_op_test = functools.partial(
            tf.nn.convolution, strides=list(self.si.strides_test), padding='VALID',
            dilations=list(self.si.dilation_rate_test), data_format=self._tf_data_format,
            name='Conv2D')
        super(Conv2DDenseSame, self).build(input_shape)
        try:
            # TF >=2.7
            self._convolution_op_train = self.convolution_op
        except AttributeError:
            # TF <=2.6
            self._convolution_op_train = self._convolution_op

    def call(self, inputs, *args, training=None, **kwargs):
        conv_op = self._convolution_op_train if training else self._convolution_op_test
        if hasattr(self, 'convolution_op'):
            # TF >=2.7
            self.convolution_op = conv_op
        else:
            # TF <=2.6
            self._convolution_op = conv_op
        self.strides = self.si.strides if training else self.si.strides_test
        self.dilation_rate = self.si.dilation_rate if training else self.si.dilation_rate_test
        preproc_layer = self.preproc_train if training else self.preproc_test
        inputs = preproc_layer(inputs) if preproc_layer is not None else inputs
        return super(Conv2DDenseSame, self).call(inputs, *args, **kwargs)

    def _make_preproc_layer(self, dilations, bottomright_stride):
        if self.kernel_size in ((3, 3), (5, 5)):
            p = dilations[0] * ((self.kernel_size[0] - 1) // 2)
            q = dilations[1] * ((self.kernel_size[1] - 1) // 2)
            if bottomright_stride:
                return keras.layers.ZeroPadding2D(((p - 1, p + 1), (q - 1, q + 1)))
            else:
                return keras.layers.ZeroPadding2D(((p, p), (q, q)))
        elif self.kernel_size == (1, 1):
            if bottomright_stride:
                return keras.layers.Cropping2D(((1, 0), (1, 0)))
            else:
                return None
        else:
            raise Exception


class TrainTestSwitchLayer(keras.layers.Layer):
    def __init__(self, train_layer, test_layer):
        super(TrainTestSwitchLayer, self).__init__()
        self.train_layer = train_layer
        self.test_layer = test_layer
        self.input_spec = self.train_layer.input_spec

    def build(self, input_shape):
        self.train_layer.build(input_shape)
        self.test_layer.build(input_shape)

    def call(self, inputs, *args, training=None, **kwargs):
        layer = self.train_layer if training else self.test_layer
        res = layer(inputs, *args, training=training, **kwargs)
        return res


class StridingInfo:
    def __init__(
            self, strides=1, strides_test=None, dilation_rate=1, dilation_rate_test=None,
            bottomright_stride=False, bottomright_stride_test=False):
        self.strides_test = conv_utils.normalize_tuple(
            strides if strides_test is None else strides_test, 2,
            'strides_test')
        self.dilation_rate_test = conv_utils.normalize_tuple(
            dilation_rate if dilation_rate_test is None else dilation_rate_test, 2,
            'dilation_rate_test')

        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.bottomright_stride = bottomright_stride
        self.bottomright_stride_test = bottomright_stride_test

        if self.bottomright_stride:
            assert self.strides == (2, 2)

        if self.bottomright_stride_test:
            assert self.strides_test == (2, 2)
