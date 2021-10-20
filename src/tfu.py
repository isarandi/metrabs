import multiprocessing as mp

import numpy as np
import tensorflow as tf

import util

_DATA_FORMAT = None
_DTYPE = None

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
    ndims = arr.shape.rank
    # convert negative indices to positive and sort descending
    axes = sorted([ax if ax >= 0 else ndims + ax + 1 for ax in axes], reverse=True)
    for ax in axes:
        arr = tf.expand_dims(arr, ax)
    return arr


def reduce_mean_masked(input_tensor, is_valid, axis=None, keepdims=False, try_fast=False):
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
    n_new_dims = input_tensor.shape.rank - is_valid.shape.rank
    is_valid = expand_dims(is_valid, [-1] * n_new_dims)
    replaced = tf.where(is_valid, input_tensor, tf.constant(0, input_tensor.dtype))
    sum_valid = tf.reduce_sum(replaced, axis=axis, keepdims=keepdims)
    n_valid = tf.math.count_nonzero(
        is_valid, axis=axis, keepdims=keepdims, dtype=input_tensor.dtype)
    return tf.math.divide_no_nan(sum_valid, n_valid)


def mean_stdev_masked(input_tensor, is_valid, items_axis, dimensions_axis, fixed_ref=None):
    if fixed_ref is not None:
        mean = fixed_ref
    else:
        mean = reduce_mean_masked(input_tensor, is_valid, axis=items_axis, keepdims=True)
    centered = input_tensor - mean

    n_new_dims = input_tensor.shape.rank - is_valid.shape.rank
    is_valid = expand_dims(is_valid, [-1] * n_new_dims)
    n_valid = tf.math.count_nonzero(
        is_valid, axis=items_axis, keepdims=True, dtype=input_tensor.dtype)

    sum_of_squared_deviations = reduce_sum_masked(
        tf.square(centered), is_valid, axis=[items_axis, dimensions_axis], keepdims=True)

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

    n_new_dims = input_tensor.shape.rank - is_valid.shape.rank
    is_valid = expand_dims(is_valid, [-1] * n_new_dims)
    replaced = tf.where(is_valid, input_tensor, tf.constant(0, input_tensor.dtype))
    return tf.reduce_sum(replaced, axis=axis, keepdims=keepdims)


def static_shape(tensor):
    return tensor.shape.as_list()


def static_n_channels(tensor):
    return static_shape(tensor)[channel_axis()]


def static_image_shape(tensor):
    if get_data_format() == 'NHWC':
        return static_shape(tensor)[1:3]
    else:
        return static_shape(tensor)[2:4]


def dynamic_batch_size(tensor):
    return tf.shape(tensor)[0]


def get_data_format():
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
        # for t, s in zip(result, output_shapes):
        #    t.set_shape(s)
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
        ndims = x.shape.rank
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
        ndims = x.shape.rank
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


def softmax(target, axis=-1):
    max_along_axis = tf.reduce_max(target, axis, keepdims=True)
    exponentiated = tf.exp(target - max_along_axis)
    denominator = tf.reduce_sum(exponentiated, axis, keepdims=True)
    return exponentiated / denominator


def soft_argmax(inp, axis):
    return decode_heatmap(softmax(inp, axis=axis), axis=axis)


def decode_heatmap(inp, axis=-1, output_coord_axis=-1):
    if not isinstance(axis, (tuple, list)):
        axis = [axis]

    # Convert negative axes to the corresponding positive index (e.g. -1 means last axis)
    heatmap_axes = [ax if ax >= 0 else inp.shape.rank + ax + 1 for ax in axis]
    result = []
    for ax in heatmap_axes:
        other_heatmap_axes = [other_ax for other_ax in heatmap_axes if other_ax != ax]
        summed_over_other_heatmap_axes = tf.reduce_sum(inp, axis=other_heatmap_axes, keepdims=True)
        coords = tf.cast(tf.linspace(0.0, 1.0, tf.shape(inp)[ax]), inp.dtype)
        decoded = tf.tensordot(summed_over_other_heatmap_axes, coords, axes=[[ax], [0]])
        result.append(tf.squeeze(tf.expand_dims(decoded, ax), heatmap_axes))
    return tf.stack(result, axis=output_coord_axis)


def decode_heatmap_with_offsets(inp, offset, axis=-1):
    if not isinstance(axis, (tuple, list)):
        axis = [axis]

    # Convert negative axes to the corresponding positive index (e.g. -1 means last axis)
    heatmap_axes = [ax if ax >= 0 else inp.shape.rank + ax + 1 for ax in axis]
    heatmap_shape = [tf.shape(inp)[ax] for ax in heatmap_axes]

    rel_coords = tf.stack(tf.meshgrid(
        *[tf.linspace(0.0, 1.0, s) for s in heatmap_shape], indexing='ij'), axis=-1)
    rel_coord_shape = [tf.shape(inp)[ax] for ax in range(inp.shape.rank)]
    rel_coord_shape.append(len(heatmap_axes))
    for ax in range(inp.shape.rank):
        if ax not in heatmap_axes:
            rel_coord_shape[ax] = 1

    rel_coords = tf.transpose(rel_coords, [*np.argsort(heatmap_axes), len(heatmap_axes)])
    rel_coords = tf.reshape(rel_coords, rel_coord_shape)
    vote_coords = tf.cast(offset, tf.float32) + rel_coords
    vote_coords = tf.clip_by_value(vote_coords, 0, 1)
    return tf.reduce_sum(inp[..., tf.newaxis] * vote_coords, axis=heatmap_axes)


def index_grid(shape):
    """Returns `len(shape)` tensors, each of shape `shape`. Each tensor contains the corresponding
    index. """
    if isinstance(shape, (list, tuple)):
        ndim = len(shape)
    else:
        ndim = static_shape(shape)[0]

    return tf.meshgrid(*[tf.range(shape[i]) for i in range(ndim)], indexing='ij')


def load_image(path, ratio=1):
    return tf.image.decode_jpeg(
        tf.io.read_file(path), fancy_upscaling=False, dct_method='INTEGER_FAST', ratio=ratio)


def resized_size_and_rest(input_shape, target_shape):
    target_shape_float = tf.cast(target_shape, tf.float32)
    input_shape_float = tf.cast(input_shape, tf.float32)
    factor = tf.reduce_min(target_shape_float / input_shape_float)
    target_shape_part = tf.cast(factor * input_shape_float, tf.int32)
    rest_shape = target_shape - target_shape_part
    return factor, target_shape_part, rest_shape


def resize_with_pad(image, target_shape):
    if image.ndim == 3:
        return tf.squeeze(resize_with_pad(image[tf.newaxis], target_shape), 0)
    factor, target_shape_part, rest_shape = resized_size_and_rest(
        tf.shape(image)[1:3], target_shape)
    if factor > 1:
        image = tf.cast(tf.image.resize(
            image, target_shape_part, method=tf.image.ResizeMethod.BILINEAR), image.dtype)
    else:
        image = tf.cast(tf.image.resize(
            image, target_shape_part, method=tf.image.ResizeMethod.AREA), image.dtype)

    return tf.pad(image, [(0, 0), (rest_shape[0], 0), (rest_shape[1], 0), (0, 0)])


def resize_with_unpad(image, orig_shape):
    if image.ndim == 3:
        return tf.squeeze(resize_with_unpad(image[tf.newaxis], orig_shape), 0)
    factor, _, rest_shape = resized_size_and_rest(orig_shape, tf.shape(image)[1:3])
    image = image[:, rest_shape[0]:, rest_shape[1]:]
    if factor < 1:
        image = tf.cast(
            tf.image.resize(
                image, orig_shape, method=tf.image.ResizeMethod.BILINEAR), image.dtype)
    else:
        image = tf.cast(
            tf.image.resize(
                image, orig_shape, method=tf.image.ResizeMethod.AREA), image.dtype)
    return image


def type_spec_from_nested(x):
    if isinstance(x, dict):
        return {k: type_spec_from_nested(v) for k, v in x.items()}
    elif isinstance(x, tuple):
        return tuple([type_spec_from_nested(v) for v in x])
    else:
        return tf.type_spec_from_value(x)


def auc(x, t1, t2):
    t1 = tf.cast(t1, tf.float32)
    t2 = tf.cast(t2, tf.float32)
    return tf.nn.relu(np.float32(1) - tf.nn.relu(x - t1) / (t2 - t1))

