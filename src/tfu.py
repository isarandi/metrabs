import functools
import logging
import multiprocessing as mp
import re

import attrdict
import numpy as np
import tensorflow as tf

import util

_IS_TRAINING = None
_DATA_FORMAT = None
_DTYPE = None

_COUNTERS = {}

TRAIN = 0
VALID = 1
TEST = 2


def expand_dims(arr, axes):
    """Inserts new dimensions of size 1 into a tensor's shape at the given positions `axes`.
    The positions are all intepreted w.r.t. the shape of `arr` as it is *now*, therefore the order
    of `axes` doesn't matter.
    Repetition of the same axis is possible and inserts multiple new dimensions
    in that position.

     0   1   2   3   4   5  <- meaning of positive numbers in `axes`
       X   X   X   X   X    <- current elements of `arr.shape`
    -6  -5  -4  -3  -2  -1  <- meaning of negative numbers in `axes`
    """
    ndims = arr.get_shape().ndims
    # convert negative indices to positive and sort descending
    axes = sorted([ax if ax >= 0 else ndims + ax + 1 for ax in axes], reverse=True)
    for ax in axes:
        arr = tf.expand_dims(arr, ax)
    return arr


def reduce_mean_masked(input_tensor, is_valid, axis=None, keepdims=False, try_fast=True):
    """Compute the mean of elements across dimensions of a tensor, ignoring elements if
    the corresponding element in `mask` is False.

    In general, `K = dim(mask) <= dim(input_tensor) = L`, and `mask`'s shape must match
    the first K dimensions of `tensor`'s shape. Then `input_tensor[i1,...,iK,...iL]` is
    ignored iff `mask[i1,...,iK]` is False.
    """
    if try_fast:
        return tf.cond(
            tf.reduce_all(is_valid),
            lambda: tf.reduce_mean(input_tensor, axis=axis, keepdims=keepdims),
            lambda: reduce_mean_masked(input_tensor, is_valid, axis, keepdims, try_fast=False))

    if axis is None and not keepdims:
        return tf.reduce_mean(tf.boolean_mask(input_tensor, is_valid))

    n_new_dims = input_tensor.get_shape().ndims - is_valid.get_shape().ndims
    is_valid = expand_dims(is_valid, [-1] * n_new_dims)
    is_valid = broadcast_like(is_valid, input_tensor)
    replaced = tf.where(is_valid, input_tensor, tf.zeros_like(input_tensor))
    sum_valid = tf.reduce_sum(replaced, axis=axis, keepdims=keepdims)
    n_valid = tf.math.count_nonzero(
        is_valid, axis=axis, keepdims=keepdims, dtype=input_tensor.dtype)
    return tf.math.divide_no_nan(sum_valid, n_valid)


def mean_stdev_masked(input_tensor, is_valid, items_axis, dimensions_axis, fixed_ref=None):
    n_new_dims = input_tensor.get_shape().ndims - is_valid.get_shape().ndims
    is_valid = expand_dims(is_valid, [-1] * n_new_dims)
    is_valid_b = broadcast_like(is_valid, input_tensor)

    if fixed_ref is not None:
        mean = fixed_ref
    else:
        mean = reduce_mean_masked(input_tensor, is_valid_b, axis=items_axis, keepdims=True)
    centered = input_tensor - mean
    n_valid = tf.math.count_nonzero(
        is_valid, axis=items_axis, keepdims=True, dtype=input_tensor.dtype)

    sum_of_squared_deviations = reduce_sum_masked(
        tf.square(centered), is_valid_b, axis=[items_axis, dimensions_axis], keepdims=True)

    stdev = tf.sqrt(tf.math.divide_no_nan(sum_of_squared_deviations, n_valid) + 1e-10)
    return mean, stdev


def reduce_sum_masked(input_tensor, is_valid, axis=None, keepdims=False):
    """Compute the mean of elements across dimensions of a tensor, ignoring elements if
    the corresponding element in `mask` is True.

    In general, `K = dim(mask) <= dim(input_tensor) = L`, and `mask`'s shape must match
    the first K dimensions of `tensor`'s shape. Then `input_tensor[i1,...,iK,...iL]` is
    ignored iff `mask[i1,...,iK]` is True.
    """
    if axis is None and not keepdims:
        return tf.reduce_sum(tf.boolean_mask(input_tensor, is_valid))

    n_new_dims = input_tensor.get_shape().ndims - is_valid.get_shape().ndims
    for i in range(n_new_dims):
        is_valid = tf.expand_dims(is_valid, -1)
    is_valid = is_valid & (tf.ones_like(input_tensor) > 0)
    is_valid = broadcast_like(is_valid, input_tensor)
    replaced = tf.where(is_valid, input_tensor, tf.zeros_like(input_tensor))

    return tf.reduce_sum(replaced, axis=axis, keepdims=keepdims)


def static_shape(tensor):
    return tensor.get_shape().as_list()


def static_n_channels(tensor):
    return static_shape(tensor)[channel_axis()]


def static_image_shape(tensor):
    if data_format() == 'NHWC':
        return static_shape(tensor)[1:3]
    else:
        return static_shape(tensor)[2:4]


def dynamic_batch_size(tensor):
    return tf.shape(tensor)[0]



def set_is_training(b):
    global _IS_TRAINING
    _IS_TRAINING = b


def is_training():
    return _IS_TRAINING


def data_format():
    return _DATA_FORMAT


def set_data_format(df):
    global _DATA_FORMAT
    _DATA_FORMAT = df


def get_dtype():
    return _DTYPE


def get_numpy_dtype():
    return {tf.float32: np.float32, tf.float16: np.float16}[_DTYPE]


def set_dtype(dtype):
    global _DTYPE
    _DTYPE = dtype


def channel_axis():
    return _DATA_FORMAT.index('C')


def image_axes():
    return _DATA_FORMAT.index('H'), _DATA_FORMAT.index('W')



def count_trainable_params():
    return sum(np.prod(static_shape(var)) for var in tf.compat.v1.trainable_variables())


def py_func_with_shapes(func, inp=None, output_types=None, output_shapes=None, name=None):
    result = tf.numpy_function(func, inp, output_types, name)
    if isinstance(output_types, (list, tuple)):
        for t, s in zip(result, output_shapes):
            t.set_shape(s)
        return tuple(result)
    else:
        assert not isinstance(result, (list, tuple))
        result.set_shape(output_shapes)
        return result



def get_shapes_and_tf_dtypes(thing):
    if not isinstance(thing, (list, tuple)):
        thing = (thing,)

    arrays = [np.asanyarray(a) for a in thing]
    tf_types = [tf.as_dtype(a.dtype) for a in arrays]
    shapes = [tf.TensorShape(a.shape) for a in arrays]
    return tuple(shapes), tuple(tf_types)


# NHWC <-> NCHW conversions
def nhwc_to_nchw(x):
    if isinstance(x, tf.Tensor):
        ndims = x.get_shape().ndims
        if ndims == 3:
            return tf.transpose(x, [2, 0, 1])
        elif ndims == 4:
            return tf.transpose(x, [0, 3, 1, 2])
        elif ndims == 2:
            return x
        else:
            raise Exception()

    if isinstance(x, list) or isinstance(x, tuple) or x.ndim == 1:
        if len(x) == 3:
            return type(x)((x[2], x[0], x[1]))
        elif len(x) == 4:
            return type(x)((x[0], x[3], x[1], x[2]))
        elif len(x) == 2:
            return x
        raise Exception()

    if x.ndim == 3:
        return np.transpose(x, [2, 0, 1])
    elif x.ndim == 4:
        return np.transpose(x, [0, 3, 1, 2])
    elif x.ndim == 2:
        return x
    else:
        raise Exception()


def nchw_to_nhwc(x):
    if isinstance(x, tf.Tensor):
        ndims = x.get_shape().ndims
        if ndims == 3:
            return tf.transpose(x, [1, 2, 0])
        elif ndims == 4:
            return tf.transpose(x, [0, 2, 3, 1])
        elif ndims == 2:
            return x
        else:
            raise Exception()

    if isinstance(x, list) or isinstance(x, tuple) or x.ndim == 1:
        if len(x) == 3:
            return type(x)((x[1], x[2], x[0]))
        elif len(x) == 4:
            return type(x)((x[0], x[2], x[3], x[1]))
        elif len(x) == 2:
            return x
        else:
            raise Exception()

    if x.ndim == 3:
        return np.transpose(x, [1, 2, 0])
    elif x.ndim == 4:
        return np.transpose(x, [0, 2, 3, 1])
    elif x.ndim == 2:
        return x
    else:
        raise Exception()


def convert_data_format(x, src_format=None, dst_format=None):
    src_format = src_format or _DATA_FORMAT
    dst_format = dst_format or _DATA_FORMAT

    if src_format == dst_format:
        return x
    elif src_format == 'NHWC':
        return nhwc_to_nchw(x)
    else:
        return nchw_to_nhwc(x)


def nhwc_to_std(x):
    return convert_data_format(x, src_format='NHWC')


def std_to_nhwc(x):
    return convert_data_format(x, dst_format='NHWC')


def nchw_to_std(x):
    return convert_data_format(x, src_format='NCHW')


def std_to_nchw(x):
    return convert_data_format(x, dst_format='NCHW')


def get_or_create_counter(name):
    global _COUNTERS
    try:
        return _COUNTERS[name]
    except KeyError:
        var = tf.compat.v1.get_local_variable(
            f'counters/{name}', shape=[], dtype=tf.int64, initializer=tf.zeros_initializer())

        counter = attrdict.AttrDict(
            name=name, var=var, reset_op=tf.compat.v1.assign(var, 0),
            increment_op=tf.compat.v1.assign_add(var, 1))
        _COUNTERS[name] = counter
        return counter


_pool = None


def get_pool(n_workers_if_uninitialized, flags=None):
    global _pool
    if _pool is None:
        if flags is None:
            import init
            flags = init.FLAGS

        ctx = mp.get_context('spawn')
        # important to use 'spawn', because 'fork' would mean the whole memory is (lazily) copied
        # then due to copy-on-write semantics, it gets duplicated when the parent changes anything
        _pool = ctx.Pool(
            n_workers_if_uninitialized, initializer=util.init_worker_process_flags,
            initargs=(flags,))

    return _pool


def gradients_with_loss_scaling(loss, variables, loss_scale=128):
    """Gradient calculation with loss scaling to improve numerical stability
    when training with float16. Gradients are typically very small and may fall out of the
    float16 range and underflow to 0. This shifts the gradients a larger scale during backprop
    and then scales them back at the end.
    """
    gradients = tf.gradients(
        loss * loss_scale, variables, gate_gradients=tf.compat.v1.train.Optimizer.GATE_GRAPH)
    return [tf.cast(g, tf.float32) / loss_scale if g is not None else None for g in gradients]


def in_variable_scope(default_name, mixed_precision=True):
    """Puts the decorated function in a TF variable scope with the provided default name.
    The function also gains two extra arguments: "scope" and "reuse" which get passed to
    tf.compat.v1.variable_scope.
    """

    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, scope=None, reuse=None, **kwargs):
            with tf.compat.v1.variable_scope(
                    scope, default_name, reuse=reuse,
                    custom_getter=mixed_precision_getter if mixed_precision else None):
                return f(*args, **kwargs)

        return decorated

    return decorator


def mixed_precision_getter(
        getter, name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=True,
        *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the compute precision."""
    # print(f'mixed prec asked for {dtype} ({name})')
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(
        name, shape, dtype=storage_dtype, initializer=initializer, regularizer=regularizer,
        trainable=trainable, *args, **kwargs)

    if storage_dtype != dtype:
        return tf.cast(variable, dtype)

    return variable


def in_name_scope(default_name):
    """Puts the decorated function in a name scope with the provided default name.
    The function also gains an extra argument "scope" which gets passed to tf.name_scope.
    """

    def wrapper(f):
        @functools.wraps(f)
        def wrapped(*args, scope=None, **kwargs):
            name = scope if scope is not None else default_name
            with tf.name_scope(name):
                return f(*args, **kwargs)

        return wrapped

    return wrapper


def broadcast_like(tensor, target_tensor):
    if tensor.dtype == tf.bool:
        return tf.logical_and(tensor, tf.ones_like(target_tensor) > 0)
    else:
        return tensor + tf.zeros_like(target_tensor, dtype=tensor.dtype)


def softmax(target, axis=-1):
    with tf.name_scope('softmax'):
        max_along_axis = tf.reduce_max(target, axis, keepdims=True)
        exponentiated = tf.exp(target - max_along_axis)
        normalizer_denominator = tf.reduce_sum(exponentiated, axis, keepdims=True)
        return exponentiated / normalizer_denominator


def soft_argmax(inp, axis):
    softmaxed = softmax(inp, axis=axis)
    return tf.stack(decode_heatmap(softmaxed, axis), axis=-1)


def decode_heatmap(inp, axis=-1):
    input_shape = inp.get_shape().as_list()
    ndims = len(input_shape)

    def relative_coords_along_axis(ax):
        grid_shape = [1] * ndims
        grid_shape[ax] = input_shape[ax]
        grid = tf.reshape(tf.linspace(0.0, 1.0, input_shape[ax]), grid_shape)
        return tf.cast(grid, inp.dtype)

    # Single axis:
    if not isinstance(axis, (tuple, list)):
        return tf.reduce_sum(relative_coords_along_axis(axis) * inp, axis=axis)

    # Multiple axes.
    # Convert negative axes to the corresponding positive index (e.g. -1 means last axis)
    heatmap_axes = [ax if ax >= 0 else ndims + ax + 1 for ax in axis]
    result = []
    for ax in heatmap_axes:
        other_heatmap_axes = [other_ax for other_ax in heatmap_axes if other_ax != ax]
        summed_over_other_axes = tf.reduce_sum(inp, axis=other_heatmap_axes, keepdims=True)
        coords = relative_coords_along_axis(ax)
        decoded = tf.reduce_sum(coords * summed_over_other_axes, axis=ax, keepdims=True)
        result.append(tf.squeeze(decoded, heatmap_axes))

    return result


def make_pretrained_weight_loader(pretrained_path, loaded_scope, checkpoint_scope, excluded_parts):
    var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                           scope=loaded_scope)
    var_dict = {v.op.name[v.op.name.index(checkpoint_scope):]: v for v in var_list}

    new_var_dicts = []
    for k, v in var_dict.items():
        if not any(excl in k for excl in excluded_parts):
            non_copy_name = re.sub(r'_copy\d*', '', k)
            for candidate_dict in new_var_dicts:
                if non_copy_name not in candidate_dict:
                    candidate_dict[non_copy_name] = v
                    break
            else:
                new_var_dicts.append({non_copy_name: v})

    savers = [tf.compat.v1.train.Saver(var_list=d) for d in new_var_dicts]
    global_init_op = tf.compat.v1.global_variables_initializer()

    def init_fn(_, sess):
        sess.run(global_init_op)
        for saver in savers:
            logging.info('Loading pretrained weights.')
            saver.restore(sess, pretrained_path)

    return init_fn


def scalar_dict_to_summary(dic):
    return tf.compat.v1.Summary(
        value=[tf.compat.v1.Summary.Value(tag=t, simple_value=v) for t, v in dic.items()])


def map_range(x, xstart, xend, ystart, yend, clip=False):
    """Maps one interval to another via linear interpolation, optionally clipping the output
    range."""

    xstart = tf.cast(xstart, x.dtype)
    xend = tf.cast(xend, x.dtype)
    ystart = tf.cast(ystart, x.dtype)
    yend = tf.cast(yend, x.dtype)
    slope = (yend - ystart) / (xend - xstart)

    y = ystart + slope * (x - xstart)
    if clip:
        ymin = tf.minimum(ystart, yend)
        ymax = tf.maximum(ystart, yend)
        return tf.clip_by_value(y, ymin, ymax)
    return y
