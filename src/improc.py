import functools
import itertools
import logging

import PIL.Image
import cv2
import imageio
import numba
import numpy as np
import pycocotools.mask

import util


def encode_mask(mask):
    return pycocotools.mask.encode(np.asfortranarray(mask.astype(np.uint8)))


decode_mask = pycocotools.mask.decode


def resize_by_factor(im, factor, interp=None):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = rounded_int_tuple([im.shape[1] * factor, im.shape[0] * factor])
    if interp is None:
        interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


@functools.lru_cache()
def get_structuring_element(shape, ksize, anchor=None):
    if not isinstance(ksize, tuple):
        ksize = (ksize, ksize)
    return cv2.getStructuringElement(shape, ksize, anchor)


def rounded_int_tuple(p):
    return tuple(np.round(p).astype(int))


def image_extents(filepath):
    """Returns the image (width, height) as a numpy array, without loading the pixel data."""

    with PIL.Image.open(filepath) as im:
        return np.asarray(im.size)


def video_extents(filepath):
    """Returns the video (width, height) as a numpy array, without loading the pixel data."""

    with imageio.get_reader(filepath, 'ffmpeg') as reader:
        return np.asarray(reader.get_meta_data()['source_size'])


def rectangle(im, pt1, pt2, color, thickness):
    cv2.rectangle(im, rounded_int_tuple(pt1), rounded_int_tuple(pt2), color, thickness)


def line(im, p1, p2, *args, **kwargs):
    if np.asarray(p1).shape[-1] != 2 or np.asarray(p2).shape[-1] != 2:
        raise Exception('Wrong dimensionality of point in line drawing')

    try:
        cv2.line(im, rounded_int_tuple(p1), rounded_int_tuple(p2), *args, **kwargs)
    except OverflowError:
        logging.warning('Overflow in rounded_int_tuple!')


def draw_box(im, box, color=(255, 0, 0), thickness=5):
    box = np.array(box)
    rectangle(im, box[:2], box[:2] + box[2:4], color, thickness)


def circle(im, center, radius, *args, **kwargs):
    cv2.circle(im, rounded_int_tuple(center), np.round(radius).astype(int), *args, **kwargs)


def draw_stick_figure(
        im, coords, joint_info, thickness=3, brightness_increase=0, color=None, joint_dots=True,
        inplace=False):
    factor = 255 if np.issubdtype(im.dtype, np.floating) else 1
    if factor != 255 or not inplace:
        result_image = (im * factor).astype(np.uint8).copy()
        result_image = np.clip(
            result_image.astype(np.float32) + brightness_increase, 0, 255).astype(np.uint8)
    else:
        result_image = im

    if color is None:
        colors = util.cycle_over_colors()
    else:
        colors = itertools.repeat(color)

    for color, (i_joint1, i_joint2) in zip(colors, joint_info.stick_figure_edges):
        relevant_coords = coords[[i_joint1, i_joint2]]
        if not np.isnan(relevant_coords).any() and not np.isclose(0, relevant_coords).any():
            line(
                result_image, coords[i_joint1], coords[i_joint2], color=color, thickness=thickness,
                lineType=cv2.LINE_AA)

    if joint_dots:
        for i_joint, joint_name in enumerate(joint_info.names):
            if not np.isnan(coords[i_joint]).any():
                circle(
                    result_image, coords[i_joint], thickness * 1.2, color=(255, 0, 0),
                    thickness=cv2.FILLED)

    return result_image


def normalize01(im, dst=None):
    if dst is None:
        result = np.empty_like(im, dtype=np.float32)
    else:
        result = dst
    result[:] = im.astype(np.float32) / np.float32(255)
    np.clip(result, np.float32(0), np.float32(1), out=result)
    return result


@numba.jit(nopython=True)
def blend_image_numba(im1, im2, im2_weight):
    return im1 * (1 - im2_weight) + im2 * im2_weight


use_libjpeg_turbo = True
if use_libjpeg_turbo:
    import jpeg4py


    def imread_jpeg(path, dst=None):
        if isinstance(path, bytes):
            path = path.decode('utf8')
        elif isinstance(path, np.str):
            path = str(path)
        path = util.ensure_absolute_path(path)
        try:
            return jpeg4py.JPEG(path).decode(dst)
        except jpeg4py.JPEGRuntimeError:
            logging.error(f'Could not load image at {path}, JPEG error.')
            raise
else:
    def imread_jpeg(path, dst=None):
        assert dst is None
        if isinstance(path, bytes):
            path = path.decode('utf8')
        elif isinstance(path, np.str):
            path = str(path)
        path = util.ensure_absolute_path(path)
        return imageio.imread(path)


@numba.jit(nopython=True)
def paste_over(im_src, im_dst, alpha, center, inplace=False):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending.

    The resulting image has the same shape as `im_dst` but contains `im_src`
    (perhaps only partially, if it's put near the border).
    Locations outside the bounds of `im_dst` are handled as expected
    (only a part or none of `im_src` becomes visible).

    Args:
        im_src: The image to be pasted onto `im_dst`. Its size can be arbitrary.
        im_dst: The target image.
        alpha: A float (0.0-1.0) image of the same size as `im_src` controlling the alpha blending
            at each pixel. Large values mean more visibility for `im_src`.
        center: coordinates in `im_dst` where the center of `im_src` should be placed.

    Returns:
        An image of the same shape as `im_dst`, with `im_src` pasted onto it.
    """

    width_height_src = np.array([im_src.shape[1], im_src.shape[0]], dtype=np.int32)
    width_height_dst = np.array([im_dst.shape[1], im_dst.shape[0]], dtype=np.int32)

    center_float = center.astype(np.float32)
    np.round(center_float, 0, center_float)
    center_int = center_float.astype(np.int32)
    ideal_start_dst = center_int - width_height_src // np.int32(2)
    ideal_end_dst = ideal_start_dst + width_height_src

    zeros = np.zeros_like(ideal_start_dst)
    start_dst = np.minimum(np.maximum(ideal_start_dst, zeros), width_height_dst)
    end_dst = np.minimum(np.maximum(ideal_end_dst, zeros), width_height_dst)

    if inplace:
        result = im_dst
    else:
        result = im_dst.copy()

    region_dst = result[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

    start_src = start_dst - ideal_start_dst
    end_src = width_height_src + (end_dst - ideal_end_dst)

    alpha_expanded = np.expand_dims(alpha, -1)
    alpha_expanded = alpha_expanded[start_src[1]:end_src[1], start_src[0]:end_src[0]]

    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]

    result[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
        (alpha_expanded * region_src + (1 - alpha_expanded) * region_dst)).astype(np.uint8)
    return result


def adjust_gamma_cuda(image, gamma, inplace=False):
    if inplace:
        return get_gamma_lookup_table_cuda(gamma).transform(image, dst=image)
    return get_gamma_lookup_table_cuda(gamma).transform(image)


def adjust_gamma(image, gamma, inplace=False):
    if isinstance(image, cv2.cuda_GpuMat):
        return adjust_gamma_cuda(image, gamma, inplace)

    if inplace:
        cv2.LUT(image, get_gamma_lookup_table(gamma), dst=image)
        return image

    return cv2.LUT(image, get_gamma_lookup_table(gamma))


@functools.lru_cache()
def get_gamma_lookup_table(gamma):
    return (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)


@functools.lru_cache()
def get_gamma_lookup_table_cuda(gamma):
    return cv2.cuda.createLookUpTable((np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8))


def blend_image(im1, im2, im2_weight):
    if im2_weight.ndim == im1.ndim - 1:
        im2_weight = im2_weight[..., np.newaxis]

    return blend_image_numba(
        im1.astype(np.float32),
        im2.astype(np.float32),
        im2_weight.astype(np.float32)).astype(im1.dtype)


@numba.jit(nopython=True)
def blend_image_numba(im1, im2, im2_weight):
    return im1 * (1 - im2_weight) + im2 * im2_weight


def is_image_readable(path):
    try:
        imread_jpeg(path)
        return True
    except:
        return False


def white_balance(img, a=None, b=None):
    result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    avg_a = a if a is not None else np.mean(result[..., 1])
    avg_b = b if b is not None else np.mean(result[..., 2])
    result[..., 1] = result[..., 1] - ((avg_a - 128) * (result[..., 0] / 255.0) * 1.1)
    result[..., 2] = result[..., 2] - ((avg_b - 128) * (result[..., 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    return result


def bb_of_mask_encoded(mask):
    return pycocotools.mask.toBbox(mask)


def mask_union_encoded(masks):
    return pycocotools.mask.merge(masks, intersect=False)


def mask_intersect_encoded(masks):
    return pycocotools.mask.merge(masks, intersect=True)


def mask_area_encoded(mask):
    return pycocotools.mask.area(mask)


def mask_subtract_encoded(mask1, mask2):
    mask2_inv = pycocotools.mask.invert(mask2)
    return mask_intersect_encoded([mask1, mask2_inv])


def largest_connected_component(mask):
    mask = mask.astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    areas = stats[1:, -1]
    if len(areas) < 1:
        return mask, np.array([0, 0, 0, 0])

    largest_area_label = 1 + np.argsort(areas)[-1]
    obj_mask = np.uint8(labels == largest_area_label)
    obj_box = stats[largest_area_label, :4]

    return obj_mask, np.array(obj_box)
