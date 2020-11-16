#!/usr/bin/env python3
import functools
import os
import os.path
import shutil

import cachetools.func
import cv2
import h5py
import numpy as np

import boxlib
import data.muco
import improc
import matlabfile
import paths
import util

matloader = functools.lru_cache(8192)(matlabfile.load)


def main():
    composite_frame_origins = save_composite_frame_origins()
    i_valids = save_valid_composite_indices(composite_frame_origins)
    create_directory_structure()

    generate_all_visible_boxes(composite_frame_origins, i_valids)
    generate_all_overall_masks(composite_frame_origins, i_valids)


def create_directory_structure():
    """Give the image files some nested directory structure, so we can browse it more easily."""
    dirname = f'{paths.DATA_ROOT}/muco/unaugmented_set_001/'
    filenames = [x for x in os.listdir(dirname) if x.endswith('.jpg')]

    for n in filenames:
        src = f'{dirname}/{n}'
        dst = f'{dirname}/{n[:2]}/{n[:4]}/{n}'
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)


def save_composite_frame_origins():
    """Save the source information (which composite consists of which original frames)
     in more readable format
     """
    f = h5py.File(f'{paths.DATA_ROOT}/muco/sampled_dataset_info.mat', 'r')
    c = f['composite_info']

    def which_frame(i_sample, i_person):
        return f[f[c[i_person, i_sample]][0, 0]][:, 0].astype(int)

    n_samples = c.shape[1]
    composite_frame_origins = np.array(
        [[which_frame(i_sample, i_person)
          for i_person in range(4)]
         for i_sample in range(n_samples)])
    np.save(f'{paths.DATA_ROOT}/muco/composite_frame_origins.npy', composite_frame_origins)
    return composite_frame_origins


def save_valid_composite_indices(composite_frame_origins):
    """Check which composite frames are actually valid
     Some will be invalid because either the RGB image or one of the masks is missing
     from the original 3DHP release
     """
    validity = [
        all(is_frame_valid(composite_frame_origins[i_sample, i_person])
            for i_person in range(4))
        for i_sample in range(composite_frame_origins.shape[0])]
    i_valids = np.nonzero(validity)[0]
    np.save(f'{paths.DATA_ROOT}/muco/valid_composite_frame_indices.npy', i_valids)
    return i_valids


def is_frame_valid(frame_info):
    i_sub, i_seq, i_cam, i_frame = frame_info
    tdhp_root = f'{paths.DATA_ROOT}/3dhp'
    im_path = f'{tdhp_root}/S{i_sub}/Seq{i_seq}/imageSequence/img_{i_cam}_{i_frame:06d}.jpg'
    fgmask_path = f'{tdhp_root}/S{i_sub}/Seq{i_seq}/FGmasks/img_{i_cam}_{i_frame:06d}.jpg'
    chair_path = f'{tdhp_root}/S{i_sub}/Seq{i_seq}/ChairMasks/img_{i_cam}_{i_frame:06d}.jpg'
    return all(os.path.exists(p) for p in [im_path, fgmask_path, chair_path])


def generate_all_visible_boxes(composites, i_valids):
    """Saves the visible bounding box for each person instance.
     Visible bounding box means the box around the unoccluded foreground mask.
     A part of the foreground mask may be occluded by a person or chair pasted in front.
     """
    result = np.zeros((composites.shape[0], 4, 4), np.float32)

    def setter(i):
        def fn(val):
            result[i] = val

        return fn

    with util.BoundedPool(None, 120) as pool:
        for i, sample in zip(i_valids, util.progressbar(composites[i_valids])):
            pool.apply_async(get_visible_boxes, (sample,), callback=setter(i))
    np.save(f'{paths.DATA_ROOT}/muco/visible_boxes.npy', result)


@functools.lru_cache()
def empty_mask():
    return improc.encode_mask(np.zeros((2048, 2048), dtype=np.uint8))


def get_visible_boxes(people):
    depth_order = np.argsort([
        root_depth(int(i_subject), int(i_seq), int(i_cam), int(i_frame))
        for i_subject, i_seq, i_cam, i_frame in people])

    def _get_mask(i):
        i_subject, i_seq, i_cam, i_frame = people[depth_order[i]]
        return get_mask(int(i_subject), int(i_seq), int(i_cam), int(i_frame))

    def _get_chair_mask(i):
        i_subject, i_seq, i_cam, i_frame = people[depth_order[i]]
        return get_chair_mask(int(i_subject), int(i_seq), int(i_cam), int(i_frame))

    boxes = np.zeros((4, 4), np.float32)
    occluder_mask = empty_mask()

    for i in range(4):
        mask = _get_mask(i)
        boxes[depth_order[i]] = improc.bb_of_mask_encoded(
            improc.mask_subtract_encoded(mask, occluder_mask))
        chair_mask = _get_chair_mask(i)
        occluder_mask = improc.mask_union_encoded([occluder_mask, mask, chair_mask])
    return boxes


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/person_coords_3dhp')
def get_coords(i_subject, i_seq, i_cam, anno_name):
    seqpath = f'{paths.DATA_ROOT}/3dhp/S{i_subject}/Seq{i_seq}'
    anno_file = matloader(f'{seqpath}/annot.mat')

    selected_joints = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 3, 6, 4]
    n_all_joints = 28

    camcoords = anno_file[anno_name][i_cam].reshape(
        [-1, n_all_joints, 3])[:, selected_joints].astype(np.float16)
    camera = data.muco.load_cameras(f'{seqpath}/camera.calibration')[i_cam]
    world_coords = np.array([camera.camera_to_world(c) for c in camcoords], dtype=np.float16)
    image_coords = np.array([camera.camera_to_image(c) for c in camcoords], dtype=np.float16)
    return world_coords, camcoords, image_coords


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/person_boxes_3dhp')
def get_box(i_subject, i_seq, i_cam, i_frame):
    imcoords = get_coords(i_subject, i_seq, i_cam, 'annot3')[2][i_frame]
    box = boxlib.expand(boxlib.bb_of_points(imcoords), 1.05)
    return boxlib.intersect(boxlib.full_box((2048, 2048)), box).astype(np.float32)


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/person_mask_3dhp')
def get_mask(i_subject, i_seq, i_cam, i_frame):
    chroma_frame = improc.imread_jpeg(
        f'{paths.DATA_ROOT}/3dhp/S{i_subject}/Seq{i_seq}/FGmasks/img_{i_cam}_{i_frame:06d}.jpg')
    person_box = get_box(i_subject, i_seq, i_cam, i_frame)

    is_fg = chroma_frame[..., 0] > 100
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        is_fg.astype(np.uint8), 4, cv2.CV_32S)
    component_boxes = stats[:, :4]
    ious = [boxlib.iou(component_box, person_box)
            for component_box in component_boxes]
    ious[0] = 0
    person_label = np.argmax(ious)
    mask = (labels == person_label).astype(np.uint8)

    # Remove foreground pixels that are far from the person box
    intbox = boxlib.intersect(
        boxlib.full_box((2048, 2048)), boxlib.expand(person_box, 1.3)).astype(int)
    mask[:intbox[1]] = 0
    mask[:, :intbox[0]] = 0
    mask[:, intbox[0] + intbox[2]:] = 0
    mask[intbox[1] + intbox[3]:] = 0
    encoded_mask = improc.encode_mask(mask)
    return encoded_mask


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/chair_mask_3dhp')
def get_chair_mask(i_subject, i_seq, i_cam, i_frame):
    p = f'{paths.DATA_ROOT}/3dhp/S{i_subject}/Seq{i_seq}/ChairMasks/img_{i_cam}_{i_frame:06d}.jpg'
    chroma_frame = improc.imread_jpeg(p)
    is_fg = chroma_frame[..., 0] < 32
    mask, objbox = improc.largest_connected_component(is_fg)
    encoded_mask = improc.encode_mask(mask)
    return encoded_mask


@cachetools.func.rr_cache(2 ** 28)
def root_depth(i_subject, i_seq, i_cam, i_frame):
    return float(get_coords(i_subject, i_seq, i_cam, 'annot3')[1][i_frame, -1, 2].copy())


def get_image(i_subj, i_seq, i_cam, i_frame):
    return improc.imread_jpeg(
        f'{paths.DATA_ROOT}/3dhp/S{i_subj}/Seq{i_seq}/imageSequence/img_{i_cam}_{i_frame:06d}.jpg')


def get_composite_image(i_sample):
    s = f'{i_sample + 1:06d}'
    return improc.imread_jpeg(f'{paths.DATA_ROOT}/muco/unaugmented_set_001/{s[:2]}/{s[:4]}/{s}.jpg')


def generate_all_overall_masks(composites, i_valids):
    with util.BoundedPool(None, 120) as pool:
        for i, sample in zip(i_valids, util.progressbar(composites[i_valids])):
            pool.apply_async(save_overall_mask, (sample, i))



def save_overall_mask(people, i_out):
    def _get_mask(i):
        i_subject, i_seq, i_cam, i_frame = people[i]
        encoded_mask = get_mask(int(i_subject), int(i_seq), int(i_cam), int(i_frame))
        return improc.decode_mask(encoded_mask)

    def _get_chair_mask(i):
        i_subject, i_seq, i_cam, i_frame = people[i]
        encoded_mask = get_chair_mask(int(i_subject), int(i_seq), int(i_cam), int(i_frame))
        return improc.decode_mask(encoded_mask)

    overall_mask = _get_mask(0)
    np.maximum(overall_mask, _get_chair_mask(0), out=overall_mask)

    for i in range(1, 4):
        np.maximum(overall_mask, _get_mask(i), out=overall_mask)
        np.maximum(overall_mask, _get_chair_mask(i), out=overall_mask)

    s = f'{i_out + 1:06d}'
    out_path = f'{paths.DATA_ROOT}/muco/masks/{s[:2]}/{s[:4]}/{s}.pkl'
    util.dump_pickle(improc.encode_mask(overall_mask), out_path)


if __name__ == '__main__':
    main()
