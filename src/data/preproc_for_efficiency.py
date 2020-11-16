"""Functions to transform (reproject, i.e. scale and crop) images as a preprocessing step.
This helps us avoid loading and decoding the full JPEG images at training time.
Instead, we just load the much smaller cropped and resized images.
"""

import copy
import functools

import cv2
import imageio
import numpy as np

import boxlib
import cameralib
import improc
import util


@functools.lru_cache()
def get_memory(shape):
    im = np.empty(shape=[*shape[:2], 3], dtype=np.float32)
    cv2.cuda.registerPageLocked(im)
    cuda_im = cv2.cuda_GpuMat(shape[0], shape[1], cv2.CV_32FC3)
    return im, cuda_im


def make_efficient_example(
        ex, new_image_path, further_expansion_factor=1, image_adjustments_3dhp=False):
    """Make example by storing the image in a cropped and resized version for efficient loading"""

    is3d = hasattr(ex, 'world_coords')
    w, h = improc.image_extents(util.ensure_absolute_path(ex.image_path))
    full_box = boxlib.full_box(imsize=[w, h])

    if is3d:
        old_camera = ex.camera
        new_camera = ex.camera.copy()
        new_camera.turn_towards(target_image_point=boxlib.center(ex.bbox))
        new_camera.undistort()
    else:
        old_camera = cameralib.Camera.create2D()
        new_camera = old_camera.copy()

    reprojected_box = reproject_box(ex.bbox, old_camera, new_camera, method='side_midpoints')
    reprojected_full_box = reproject_box(full_box, old_camera, new_camera, method='corners')
    expanded_bbox = get_expanded_crop_box(
        reprojected_box, reprojected_full_box, further_expansion_factor)
    scale_factor = min(1.2, 256 / np.max(reprojected_box[2:]) * 1.5)
    new_camera.shift_image(-expanded_bbox[:2])
    new_camera.scale_output(scale_factor)

    reprojected_box = reproject_box(ex.bbox, old_camera, new_camera, method='side_midpoints')
    dst_shape = improc.rounded_int_tuple(scale_factor * expanded_bbox[[3, 2]])

    new_image_abspath = util.ensure_absolute_path(new_image_path)
    if not (util.is_file_newer(new_image_abspath, "2020-10-03T23:00:00")
                    and improc.is_image_readable(new_image_abspath)):
        im = improc.imread_jpeg(ex.image_path)
        host_im, cuda_im = get_memory(im.shape)
        np.power((im.astype(np.float32) / 255), 2.2, out=host_im)
        cuda_im.upload(host_im)
        new_im = cameralib.reproject_image(
            cuda_im, old_camera, new_camera, dst_shape, antialias_factor=2, interp=cv2.INTER_CUBIC)
        new_im = np.clip(new_im.download(), 0, 1)

        if image_adjustments_3dhp:
            # enhance the 3dhp images to reduce the green tint and increase brightness
            new_im = (new_im ** (1 / 2.2 * 0.67) * 255).astype(np.uint8)
            new_im = improc.white_balance(new_im, 110, 145)
        else:
            new_im = (new_im ** (1 / 2.2) * 255).astype(np.uint8)
        util.ensure_path_exists(new_image_abspath)
        imageio.imwrite(new_image_abspath, new_im, quality=95)

    new_ex = copy.deepcopy(ex)
    new_ex.bbox = reprojected_box
    new_ex.image_path = new_image_path
    if is3d:
        new_ex.camera = new_camera
    else:
        new_ex.coords = cameralib.reproject_image_points(new_ex.coords, old_camera, new_camera)

    if hasattr(ex, 'mask') and ex.mask is not None:
        if isinstance(ex.mask, str):
            mask = improc.imread_jpeg(util.ensure_absolute_path(ex.mask))
            host_mask, cuda_mask = get_memory(mask.shape)
            np.divide(mask.astype(np.float32), 255, out=host_mask)
            cuda_mask.upload(host_mask)
            mask_reproj = cameralib.reproject_image(
                cuda_mask, ex.camera, new_camera, dst_shape, antialias_factor=2).download()
            mask_reproj = 255 * (mask_reproj[..., 0] > 32 / 255).astype(np.uint8)
            new_ex.mask = get_connected_component_with_highest_iou(mask_reproj, reprojected_box)
        else:
            new_ex.mask = ex.mask
    return new_ex


def reproject_box(old_box, old_camera, new_camera, method='balanced'):
    center = boxlib.center(old_box)
    dx = np.array([old_box[2] / 2, 0])
    dy = np.array([0, old_box[3] / 2])
    new_midpoint_box = None
    new_corner_box = None

    if method in ('balanced', 'side_midpoints'):
        old_side_midpoints = center + np.stack([-dx, -dy, dx, dy])
        new_side_midpoints = cameralib.reproject_image_points(
            old_side_midpoints, old_camera, new_camera)
        new_midpoint_box = boxlib.bb_of_points(new_side_midpoints)

    if method in ('balanced', 'corners'):
        old_corners = center + np.stack([-dx - dy, dx - dy, dx + dy, -dx + dy])
        new_corners = cameralib.reproject_image_points(
            old_corners, old_camera, new_camera)
        new_corner_box = boxlib.bb_of_points(new_corners)

    if method == 'corners':
        return new_corner_box
    elif method == 'side_midpoints':
        return new_midpoint_box
    elif method == 'balanced':
        return np.mean([new_midpoint_box, new_corner_box], axis=0)
    else:
        raise ValueError


def get_expanded_crop_box(bbox, full_box, further_expansion_factor):
    max_rotate = np.pi / 6
    padding_factor = 1 / 0.85
    scale_down_factor = 1 / 0.85
    shift_factor = 1.1
    s, c = np.sin(max_rotate), np.cos(max_rotate)
    w, h = bbox[2:]
    box_center = boxlib.center(bbox)
    rot_bbox_side = max(c * w + s * h, c * h + s * w)
    rot_bbox = boxlib.box_around(box_center, rot_bbox_side)
    expansion_factor = (
            padding_factor * shift_factor * scale_down_factor * further_expansion_factor)
    expanded_bbox = boxlib.intersect(
        boxlib.expand(rot_bbox, expansion_factor), full_box)
    return expanded_bbox


def get_connected_component_with_highest_iou(mask, person_box):
    """Finds the 4-connected component in `mask` with the highest bbox IoU with the `person box`"""
    mask = mask.astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    component_boxes = stats[:, :4]
    ious = [boxlib.iou(component_box, person_box) for component_box in component_boxes]
    person_label = np.argmax(ious)
    return improc.encode_mask(labels == person_label)
