# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains building blocks for various versions of Residual Networks.

Residual networks (ResNets) were proposed in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition. arXiv:1512.03385, 2015

More variants were introduced in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks. arXiv: 1603.05027, 2016

We can obtain different ResNet variants by changing the network depth, width,
and form of residual unit. This module implements the infrastructure for
building them. Concrete ResNet units and full ResNet networks are implemented in
the accompanying resnet_v1.py and resnet_v2.py modules.

Compared to https://github.com/KaimingHe/deep-residual-networks, in the current
implementation we subsample the output activations in the last residual unit of
each block, instead of subsampling the input activations in the first residual
unit of each block. The two implementations give identical results but our
implementation is more memory efficient.
"""

from __future__ import absolute_import, division, print_function

import collections

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, nn_ops, variable_scope
from tf_slim import layers as layers_lib
from tf_slim.layers import initializers, layers, regularizers, utils
from tf_slim.ops.arg_scope import add_arg_scope, arg_scope

import tfu
from options import FLAGS


# pylint:enable=g-direct-tensorflow-import


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.

    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ResNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ResNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride) tuple for each unit in the
        block to serve as argument to unit_fn.
    """


def subsample(inputs, factor, scope=None):
    """Subsamples the input along the spatial dimensions.

    Args:
      inputs: A `Tensor` of size [batch, height_in, width_in, channels].
      factor: The subsampling factor.
      scope: Optional variable_scope.

    Returns:
      output: A `Tensor` of size [batch, height_out, width_out, channels] with the
        input, either intact (if factor == 1) or subsampled (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        return layers.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(
        inputs, num_outputs, kernel_size, stride=1, rate=1, centered_stride=False, scope=None,
        **kwargs):
    """Strided 2-D convolution with 'SAME' padding.

    When stride > 1, then we do explicit zero-padding, followed by conv2d with
    'VALID' padding.

    Note that

       net = conv2d_same(inputs, num_outputs, 3, stride=stride)

    is equivalent to

       net = tf.contrib.layers.conv2d(inputs, num_outputs, 3, stride=1,
       padding='SAME')
       net = subsample(net, factor=stride)

    whereas

       net = tf.contrib.layers.conv2d(inputs, num_outputs, 3, stride=stride,
       padding='SAME')

    is different when the input's height or width is even, which is why we add the
    current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

    Args:
      inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
      num_outputs: An integer, the number of output filters.
      kernel_size: An int with the kernel_size of the filters.
      stride: An integer, the output stride.
      rate: An integer, rate for atrous convolution.
      scope: Scope.

    Returns:
      output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
    """
    if stride == 1 or centered_stride:
        return layers_lib.conv2d(
            inputs, num_outputs, kernel_size, stride=stride, rate=rate, padding='SAME', scope=scope,
            **kwargs)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        if tfu.data_format() == 'NHWC':
            inputs = array_ops.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        else:
            inputs = array_ops.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
        return layers_lib.conv2d(
            inputs, num_outputs, kernel_size, stride=stride, rate=rate, padding='VALID',
            scope=scope, **kwargs)


def max_pool2d_same(
        inputs, kernel_size, stride, centered_stride=False, scope=None):
    """Strided 2-D max pool with 'SAME' padding.

    When stride > 1, then we do explicit zero-padding, followed by max_pool2d with
    'VALID' padding.

    Note that

       net = max_pool2d_same(inputs, num_outputs, 3, stride=stride)

    is equivalent to

       net = tf.contrib.layers.max_pool2d(inputs, num_outputs, 3, stride=1,
       padding='SAME')
       net = subsample(net, factor=stride)

    whereas

       net = tf.contrib.layers.max_pool2d(inputs, num_outputs, 3, stride=stride,
       padding='SAME')

    is different when the input's height or width is even, which is why we add the
    current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

    Args:
      inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
      kernel_size: An int with the kernel_size of the filters.
      stride: An integer, the output stride.
      scope: Scope.

    Returns:
      output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
    """
    if stride == 1 or centered_stride:
        return layers_lib.max_pool2d(
            inputs, kernel_size, stride=stride, padding='SAME', scope=scope)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        if tfu.data_format() == 'NHWC':
            inputs = array_ops.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        else:
            inputs = array_ops.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
        return layers_lib.max_pool2d(
            inputs, kernel_size, stride=stride, padding='VALID', scope=scope)


# @add_arg_scope
# def stack_blocks_dense(net,
#                        blocks,
#                        output_stride=None,
#                        outputs_collections=None):
#     """Stacks ResNet `Blocks` and controls output feature density.
#
#     First, this function creates scopes for the ResNet in the form of
#     'block_name/unit_1', 'block_name/unit_2', etc.
#
#     Second, this function allows the user to explicitly control the ResNet
#     output_stride, which is the ratio of the input to output spatial resolution.
#     This is useful for dense prediction tasks such as semantic segmentation or
#     object detection.
#
#     Most ResNets consist of 4 ResNet blocks and subsample the activations by a
#     factor of 2 when transitioning between consecutive ResNet blocks. This results
#     to a nominal ResNet output_stride equal to 8. If we set the output_stride to
#     half the nominal network stride (e.g., output_stride=4), then we compute
#     responses twice.
#
#     Control of the output feature density is implemented by atrous convolution.
#
#     Args:
#       net: A `Tensor` of size [batch, height, width, channels].
#       blocks: A list of length equal to the number of ResNet `Blocks`. Each
#         element is a ResNet `Block` object describing the units in the `Block`.
#       output_stride: If `None`, then the output will be computed at the nominal
#         network stride. If output_stride is not `None`, it specifies the requested
#         ratio of input to output spatial resolution, which needs to be equal to
#         the product of unit strides from the start up to some level of the ResNet.
#         For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
#         then valid values for the output_stride are 1, 2, 6, 24 or None (which
#         is equivalent to output_stride=24).
#       outputs_collections: Collection to add the ResNet block outputs.
#
#     Returns:
#       net: Output tensor with stride equal to the specified output_stride.
#
#     Raises:
#       ValueError: If the target output_stride is not valid.
#     """
#     # The current_stride variable keeps track of the effective stride of the
#     # activations. This allows us to invoke atrous convolution whenever applying
#     # the next residual unit would result in the activations having stride larger
#     # than the target output_stride.
#     current_stride = 1
#
#     # The atrous convolution rate parameter.
#     rate = 1
#
#     for block in blocks:
#         with variable_scope.variable_scope(block.scope, 'block', [net]) as sc:
#             for i, unit in enumerate(block.args):
#                 if output_stride is not None and current_stride > output_stride:
#                     raise ValueError('The target output_stride cannot be reached.')
#
#                 with variable_scope.variable_scope('unit_%d' % (i + 1), values=[net]):
#                     # If we have reached the target output_stride, then we need to employ
#                     # atrous convolution with stride=1 and multiply the atrous rate by the
#                     # current unit's stride for use in subsequent layers.
#                     if output_stride is not None and current_stride == output_stride:
#                         net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
#                         rate *= unit.get('stride', 1)
#                     else:
#                         net = block.unit_fn(net, rate=1, **unit)
#                         current_stride *= unit.get('stride', 1)
#             net = utils.collect_named_outputs(outputs_collections, sc.name, net)
#
#     if output_stride is not None and current_stride != output_stride:
#         raise ValueError('The target output_stride cannot be reached.')
#
#     return net


@add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None,
                       store_non_strided_activations=False,
                       outputs_collections=None):
    """Stacks ResNet `Blocks` and controls output feature density.
    First, this function creates scopes for the ResNet in the form of
    'block_name/unit_1', 'block_name/unit_2', etc.
    Second, this function allows the user to explicitly control the ResNet
    output_stride, which is the ratio of the input to output spatial resolution.
    This is useful for dense prediction tasks such as semantic segmentation or
    object detection.
    Most ResNets consist of 4 ResNet blocks and subsample the activations by a
    factor of 2 when transitioning between consecutive ResNet blocks. This results
    to a nominal ResNet output_stride equal to 8. If we set the output_stride to
    half the nominal network stride (e.g., output_stride=4), then we compute
    responses twice.
    Control of the output feature density is implemented by atrous convolution.
    Args:
      net: A `Tensor` of size [batch, height, width, channels].
      blocks: A list of length equal to the number of ResNet `Blocks`. Each
        element is a ResNet `Block` object describing the units in the `Block`.
      output_stride: If `None`, then the output will be computed at the nominal
        network stride. If output_stride is not `None`, it specifies the requested
        ratio of input to output spatial resolution, which needs to be equal to
        the product of unit strides from the start up to some level of the ResNet.
        For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
        then valid values for the output_stride are 1, 2, 6, 24 or None (which
        is equivalent to output_stride=24).
      store_non_strided_activations: If True, we compute non-strided (undecimated)
        activations at the last unit of each block and store them in the
        `outputs_collections` before subsampling them. This gives us access to
        higher resolution intermediate activations which are useful in some
        dense prediction problems but increases 4x the computation and memory cost
        at the last unit of each block.
      outputs_collections: Collection to add the ResNet block outputs.
    Returns:
      net: Output tensor with stride equal to the specified output_stride.
    Raises:
      ValueError: If the target output_stride is not valid.
    """
    # The current_stride variable keeps track of the effective stride of the
    # activations. This allows us to invoke atrous convolution whenever applying
    # the next residual unit would result in the activations having stride larger
    # than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    for block in blocks:
        with variable_scope.variable_scope(block.scope, 'block', [net]) as sc:
            block_stride = 1
            for i, unit in enumerate(block.args):
                if store_non_strided_activations and i == len(block.args) - 1:
                    # Move stride from the block's last unit to the end of the block.
                    block_stride = unit.get('stride', 1)
                    unit = dict(unit, stride=1)

                with variable_scope.variable_scope('unit_%d' % (i + 1), values=[net]):
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)

                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)
                        if output_stride is not None and current_stride > output_stride:
                            raise ValueError('The target output_stride cannot be reached.')

            # Collect activations at the block's end before performing subsampling.
            net = utils.collect_named_outputs(outputs_collections, sc.name, net)

            # Subsampling of the block's output activations.
            if output_stride is not None and current_stride == output_stride:
                rate *= block_stride
            else:
                net = subsample(net, block_stride)
                current_stride *= block_stride
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net


@add_arg_scope
def stack_blocks_dense_split(
        net, blocks, n_branches=1, split_at_block=3,
        output_stride=None, store_non_strided_activations=False, outputs_collections=None):
    """Stacks ResNet `Blocks` and controls output feature density.
    First, this function creates scopes for the ResNet in the form of
    'block_name/unit_1', 'block_name/unit_2', etc.
    Second, this function allows the user to explicitly control the ResNet
    output_stride, which is the ratio of the input to output spatial resolution.
    This is useful for dense prediction tasks such as semantic segmentation or
    object detection.
    Most ResNets consist of 4 ResNet blocks and subsample the activations by a
    factor of 2 when transitioning between consecutive ResNet blocks. This results
    to a nominal ResNet output_stride equal to 8. If we set the output_stride to
    half the nominal network stride (e.g., output_stride=4), then we compute
    responses twice.
    Control of the output feature density is implemented by atrous convolution.
    Args:
      net: A `Tensor` of size [batch, height, width, channels].
      blocks: A list of length equal to the number of ResNet `Blocks`. Each
        element is a ResNet `Block` object describing the units in the `Block`.
      output_stride: If `None`, then the output will be computed at the nominal
        network stride. If output_stride is not `None`, it specifies the requested
        ratio of input to output spatial resolution, which needs to be equal to
        the product of unit strides from the start up to some level of the ResNet.
        For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
        then valid values for the output_stride are 1, 2, 6, 24 or None (which
        is equivalent to output_stride=24).
      store_non_strided_activations: If True, we compute non-strided (undecimated)
        activations at the last unit of each block and store them in the
        `outputs_collections` before subsampling them. This gives us access to
        higher resolution intermediate activations which are useful in some
        dense prediction problems but increases 4x the computation and memory cost
        at the last unit of each block.
      outputs_collections: Collection to add the ResNet block outputs.
    Returns:
      net: Output tensor with stride equal to the specified output_stride.
    Raises:
      ValueError: If the target output_stride is not valid.
    """
    # The current_stride variable keeps track of the effective stride of the
    # activations. This allows us to invoke atrous convolution whenever applying
    # the next residual unit would result in the activations having stride larger
    # than the target output_stride.
    current_strides = [1]

    # The atrous convolution rate parameter.
    rates = [1]
    nets = [net]

    for i_block, block in enumerate(blocks):
        if i_block == split_at_block:
            # We make the split here from a single "net" to multiple branches
            current_strides, nets, rates = zip(*[resnet_block_fn(
                block, current_strides[0], nets[0], output_stride, outputs_collections, rates[0],
                store_non_strided_activations, suffix('preact', i_branch))
                for i_branch in range(n_branches)])
        else:
            # Otherwise just push each "net" through a separate copy of the current block
            current_strides, nets, rates = zip(*[resnet_block_fn(
                block, current_stride, net, output_stride, outputs_collections, rate,
                store_non_strided_activations, suffix('logits', i_branch))
                for i_branch, (net, current_stride, rate) in
                enumerate(zip(nets, current_strides, rates))])

    if output_stride is not None and current_strides[0] != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return nets


def suffix(name, num):
    if FLAGS.compatibility_mode:
        return f'{name}_copy{num}'
    else:
        return name


def resnet_block_fn(
        block, current_stride, net, output_stride, outputs_collections, rate,
        store_non_strided_activations, scope_suffix):
    with variable_scope.variable_scope(block.scope, f'block{scope_suffix}', [net]) as sc:
        block_stride = 1
        for i, unit in enumerate(block.args):
            if store_non_strided_activations and i == len(block.args) - 1:
                # Move stride from the block's last unit to the end of the block.
                block_stride = unit.get('stride', 1)
                unit = dict(unit, stride=1)

            with variable_scope.variable_scope('unit_%d' % (i + 1), values=[net]):
                # If we have reached the target output_stride, then we need to employ
                # atrous convolution with stride=1 and multiply the atrous rate by the
                # current unit's stride for use in subsequent layers.
                if output_stride is not None and current_stride == output_stride:
                    net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                    rate *= unit.get('stride', 1)

                else:
                    net = block.unit_fn(net, rate=1, **unit)
                    current_stride *= unit.get('stride', 1)
                    if output_stride is not None and current_stride > output_stride:
                        raise ValueError('The target output_stride cannot be reached.')

        # Collect activations at the block's end before performing subsampling.
        net = utils.collect_named_outputs(outputs_collections, sc.name, net)

        # Subsampling of the block's output activations.
        if output_stride is not None and current_stride == output_stride:
            rate *= block_stride
        else:
            net = subsample(net, block_stride)
            current_stride *= block_stride
            if output_stride is not None and current_stride > output_stride:
                raise ValueError('The target output_stride cannot be reached.')
    return current_stride, net, rate


def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    """Defines the default ResNet arg scope.

    TODO(gpapan): The batch-normalization related default values above are
      appropriate for use in conjunction with the reference ResNet models
      released at https://github.com/KaimingHe/deep-residual-networks. When
      training ResNets from scratch, they might need to be tuned.

    Args:
      weight_decay: The weight decay to use for regularizing the model.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      batch_norm_epsilon: Small constant to prevent division by zero when
        normalizing activations by their variance in batch normalization.
      batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
        activations in the batch normalization layer.

    Returns:
      An `arg_scope` to use for the resnet models.
    """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': ops.GraphKeys.UPDATE_OPS,
    }

    with arg_scope(
            [layers_lib.conv2d],
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            weights_initializer=initializers.variance_scaling_initializer(),
            activation_fn=nn_ops.relu,
            normalizer_fn=layers.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([layers.batch_norm], **batch_norm_params):
            # The following implies padding='SAME' for pool1, which makes feature
            # alignment easier for dense prediction tasks. This is also used in
            # https://github.com/facebook/fb.resnet.torch. However the accompanying
            # code of 'Deep Residual Learning for Image Recognition' uses
            # padding='VALID' for pool1. You can switch to that choice by setting
            # tf.contrib.framework.arg_scope([tf.contrib.layers.max_pool2d], padding='VALID').
            with arg_scope([layers.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc
