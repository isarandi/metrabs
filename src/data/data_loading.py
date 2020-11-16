"""Functions for loading learning examples from disk and numpy arrays into tensors.
Augmentations are also called from here.
"""
import re

import cv2
import numpy as np

import augmentation.appearance
import augmentation.background
import augmentation.voc_loader
import boxlib
import cameralib
import improc
import tfu
import util
from options import FLAGS
from tfu import TRAIN


def load_and_transform3d(ex, joint_info, learning_phase, rng=None):
    appearance_rng = util.new_rng(rng)
    background_rng = util.new_rng(rng)
    geom_rng = util.new_rng(rng)
    partial_visi_rng = util.new_rng(rng)

    output_side = FLAGS.proc_side
    output_imshape = (output_side, output_side)

    box = ex.bbox
    if FLAGS.partial_visibility:
        box = util.random_partial_subbox(boxlib.expand_to_square(box), partial_visi_rng)

    crop_side = np.max(box[2:])
    center_point = boxlib.center(box)
    if ((learning_phase == TRAIN and FLAGS.geom_aug) or
            (learning_phase != TRAIN and FLAGS.test_aug and FLAGS.geom_aug)):
        center_point += util.random_uniform_disc(geom_rng) * FLAGS.shift_aug / 100 * crop_side

    if box[2] < box[3]:
        delta_y = np.array([0, box[3] / 2])
        sidepoints = center_point + np.stack([-delta_y, delta_y])
    else:
        delta_x = np.array([box[2] / 2, 0])
        sidepoints = center_point + np.stack([-delta_x, delta_x])

    cam = ex.camera.copy()
    cam.turn_towards(target_image_point=center_point)
    cam.undistort()
    cam.square_pixels()
    world_sidepoints = ex.camera.image_to_world(sidepoints)
    cam_sidepoints = cam.world_to_image(world_sidepoints)
    crop_side = np.linalg.norm(cam_sidepoints[0] - cam_sidepoints[1])
    cam.zoom(output_side / crop_side)
    cam.center_principal_point(output_imshape)

    if FLAGS.geom_aug and (learning_phase == TRAIN or FLAGS.test_aug):
        s1 = FLAGS.scale_aug_down / 100
        s2 = FLAGS.scale_aug_up / 100
        r = FLAGS.rot_aug * np.pi / 180
        zoom = geom_rng.uniform(1 - s1, 1 + s2)
        cam.zoom(zoom)
        cam.rotate(roll=geom_rng.uniform(-r, r))

    world_coords = ex.univ_coords if FLAGS.universal_skeleton else ex.world_coords
    metric_world_coords = ex.world_coords

    if learning_phase == TRAIN and geom_rng.rand() < 0.5:
        cam.horizontal_flip()
        camcoords = cam.world_to_camera(world_coords)[joint_info.mirror_mapping]
        metric_world_coords = metric_world_coords[joint_info.mirror_mapping]
    else:
        camcoords = cam.world_to_camera(world_coords)

    imcoords = cam.world_to_image(metric_world_coords)

    image_path = util.ensure_absolute_path(ex.image_path)
    origsize_im = improc.imread_jpeg(image_path)

    interp_str = (FLAGS.image_interpolation_train
                  if learning_phase == TRAIN else FLAGS.image_interpolation_test)
    antialias = (FLAGS.antialias_train if learning_phase == TRAIN else FLAGS.antialias_test)
    interp = getattr(cv2, 'INTER_' + interp_str.upper())
    im = cameralib.reproject_image(
        origsize_im, ex.camera, cam, output_imshape, antialias_factor=antialias, interp=interp)

    if re.match('.+/mupots/TS[1-5]/.+', ex.image_path):
        im = improc.adjust_gamma(im, 0.67, inplace=True)
    elif '3dhp' in ex.image_path and re.match('.+/(TS[1-4])/', ex.image_path):
        im = improc.adjust_gamma(im, 0.67, inplace=True)
        im = improc.white_balance(im, 110, 145)

    if (FLAGS.background_aug_prob and hasattr(ex, 'mask') and ex.mask is not None and
            background_rng.rand() < FLAGS.background_aug_prob and
            (learning_phase == TRAIN or FLAGS.test_aug)):
        fgmask = improc.decode_mask(ex.mask)
        fgmask = cameralib.reproject_image(
            fgmask, ex.camera, cam, output_imshape, antialias_factor=antialias, interp=interp)
        im = augmentation.background.augment_background(im, fgmask, background_rng)

    im = augmentation.appearance.augment_appearance(im, learning_phase, appearance_rng)
    im = tfu.nhwc_to_std(im)
    im = improc.normalize01(im)

    # Joints with NaN coordinates are invalid
    is_joint_in_fov = ~np.logical_or(np.any(imcoords < 0, axis=-1),
                                     np.any(imcoords >= FLAGS.proc_side, axis=-1))
    joint_validity_mask = ~np.any(np.isnan(camcoords), axis=-1)

    rot_to_orig_cam = ex.camera.R @ cam.R.T
    rot_to_world = cam.R.T
    inv_intrinsics = np.linalg.inv(cam.intrinsic_matrix)

    return (
        ex.image_path, im, np.nan_to_num(camcoords).astype(np.float32),
        np.nan_to_num(imcoords).astype(np.float32), inv_intrinsics.astype(np.float32),
        rot_to_orig_cam.astype(np.float32), rot_to_world.astype(np.float32),
        cam.t.astype(np.float32), joint_validity_mask,
        np.float32(is_joint_in_fov), ex.activity_name, ex.scene_name)


def load_and_transform2d(example, joint_info, learning_phase, rng):
    # Get the random number generators for the different augmentations to make it reproducibile
    appearance_rng = util.new_rng(rng)
    geom_rng = util.new_rng(rng)
    partial_visi_rng = util.new_rng(rng)

    # Load the image
    image_path = util.ensure_absolute_path(example.image_path)
    im_from_file = improc.imread_jpeg(image_path)

    # Determine bounding box
    bbox = example.bbox
    if FLAGS.partial_visibility:
        bbox = util.random_partial_subbox(boxlib.expand_to_square(bbox), partial_visi_rng)

    crop_side = np.max(bbox)
    center_point = boxlib.center(bbox)
    orig_cam = cameralib.Camera.create2D(im_from_file.shape)
    cam = orig_cam.copy()
    cam.zoom(FLAGS.proc_side / crop_side)

    if FLAGS.geom_aug:
        center_point += util.random_uniform_disc(geom_rng) * FLAGS.shift_aug / 100 * crop_side
        s1 = FLAGS.scale_aug_down / 100
        s2 = FLAGS.scale_aug_up / 100
        cam.zoom(geom_rng.uniform(1 - s1, 1 + s2))
        r = FLAGS.rot_aug * np.pi / 180
        cam.rotate(roll=geom_rng.uniform(-r, r))

    if FLAGS.geom_aug and geom_rng.rand() < 0.5:
        # Horizontal flipping
        cam.horizontal_flip()
        # Must also permute the joints to exchange e.g. left wrist and right wrist!
        imcoords = example.coords[joint_info.mirror_mapping]
    else:
        imcoords = example.coords

    new_center_point = cameralib.reproject_image_points(center_point, orig_cam, cam)
    cam.shift_to_center(new_center_point, (FLAGS.proc_side, FLAGS.proc_side))

    is_annotation_invalid = (np.nan_to_num(imcoords[:, 1]) > im_from_file.shape[0] * 0.95)
    imcoords[is_annotation_invalid] = np.nan
    imcoords = cameralib.reproject_image_points(imcoords, orig_cam, cam)

    interp_str = (FLAGS.image_interpolation_train
                  if learning_phase == TRAIN else FLAGS.image_interpolation_test)
    antialias = (FLAGS.antialias_train if learning_phase == TRAIN else FLAGS.antialias_test)
    interp = getattr(cv2, 'INTER_' + interp_str.upper())
    im = cameralib.reproject_image(
        im_from_file, orig_cam, cam, (FLAGS.proc_side, FLAGS.proc_side),
        antialias_factor=antialias, interp=interp)
    im = augmentation.appearance.augment_appearance(im, learning_phase, appearance_rng)
    im = tfu.nhwc_to_std(im)
    im = improc.normalize01(im)

    joint_validity_mask = ~np.any(np.isnan(imcoords), axis=1)
    # We must eliminate NaNs because some TensorFlow ops can't deal with any NaNs touching them,
    # even if they would not influence the result. Therefore we use a separate "joint_validity_mask"
    # to indicate which joint coords are valid.
    imcoords = np.nan_to_num(imcoords)
    return example.image_path, np.float32(im), np.float32(imcoords), joint_validity_mask
