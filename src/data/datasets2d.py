import functools

import logging
import numpy as np
import os
import os.path
import scipy.optimize

import boxlib
import matlabfile
import paths
import util
from data.joint_info import JointInfo
from data.preproc_for_efficiency import make_efficient_example
from util import TEST, TRAIN, VALID


class Pose2DDataset:
    def __init__(
            self, joint_info, train_examples=None, valid_examples=None, test_examples=None):
        self.joint_info = joint_info
        self.examples = {
            TRAIN: train_examples or [], VALID: valid_examples or [], TEST: test_examples or []}


class Pose2DExample:
    def __init__(self, image_path, coords, bbox=None):
        self.image_path = image_path
        self.coords = coords
        self.bbox = bbox


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/mpii.pkl', min_time="2020-11-04T16:46:25")
def make_mpii():
    joint_names = 'rank,rkne,rhip,lhip,lkne,lank,pelv,thor,neck,head,rwri,relb,rsho,lsho,lelb,lwri'
    edges = 'lsho-lelb-lwri,rsho-relb-rwri,lhip-lkne-lank,rhip-rkne-rank,neck-head,pelv-thor'
    joint_info_full = JointInfo(joint_names, edges)

    joint_names_used = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri'
    joint_info_used = JointInfo(joint_names_used, edges)
    dataset = Pose2DDataset(joint_info_used)
    selected_joints = [joint_info_full.ids[name] for name in joint_info_used.names]

    mat_path = f'{paths.DATA_ROOT}/mpii/mpii_human_pose_v1_u12_1.mat'
    s = matlabfile.load(mat_path).RELEASE
    annolist = np.atleast_1d(s.annolist)
    pool = util.BoundedPool(None, 120)

    for anno, is_train, rect_ids in zip(annolist, util.progressbar(s.img_train), s.single_person):
        if not is_train:
            continue

        image_path = f'mpii/images/{anno.image.name}'
        annorect = np.atleast_1d(anno.annorect)
        rect_ids = np.atleast_1d(rect_ids) - 1

        for rect_id in rect_ids:
            rect = annorect[rect_id]
            if 'annopoints' not in rect or len(rect.annopoints) == 0:
                continue

            coords = np.full(
                shape=[joint_info_full.n_joints, 2], fill_value=np.nan, dtype=np.float32)
            for joint in np.atleast_1d(rect.annopoints.point):
                coords[joint.id] = [joint.x, joint.y]

            coords = coords[selected_joints]
            rough_person_center = np.float32([rect.objpos.x, rect.objpos.y])
            rough_person_size = rect.scale * 200

            # Shift person center down like [Sun et al. 2018], who say this is common on MPII
            rough_person_center[1] += 0.075 * rough_person_size

            topleft = np.array(rough_person_center) - np.array(rough_person_size) / 2
            bbox = np.array([topleft[0], topleft[1], rough_person_size, rough_person_size])
            ex = Pose2DExample(image_path, coords, bbox=bbox)
            new_im_path = image_path.replace('mpii', 'mpii_downscaled')
            without_ext, ext = os.path.splitext(new_im_path)
            new_im_path = f'{without_ext}_{rect_id:02d}{ext}'
            pool.apply_async(
                make_efficient_example, (ex, new_im_path), callback=dataset.examples[TRAIN].append)

    print('Waiting for tasks...')
    pool.close()
    pool.join()
    print('Done...')
    dataset.examples[TRAIN].sort(key=lambda x: x.image_path)
    return dataset


@util.cache_result_on_disk(
    f'{paths.CACHE_DIR}/mpii_yolo.pkl', min_time="2020-11-02T21:39:35")
def make_mpii_yolo():
    joint_info_full = JointInfo(
        'rank,rkne,rhip,lhip,lkne,lank,pelv,thor,neck,head,rwri,relb,rsho,lsho,lelb,lwri',
        'lsho-lelb-lwri,rsho-relb-rwri,lhip-lkne-lank,rhip-rkne-rank,neck-head,pelv-thor')
    joint_info_used = JointInfo(
        'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,lelb,lwri',
        'lelb-lwri,relb-rwri,lhip-lkne-lank,rhip-rkne-rank')
    selected_joints = [joint_info_full.ids[name] for name in joint_info_used.names]

    mat_path = f'{paths.DATA_ROOT}/mpii/mpii_human_pose_v1_u12_1.mat'
    s = matlabfile.load(mat_path).RELEASE
    annolist = np.atleast_1d(s.annolist)
    all_boxes = util.load_pickle(f'{paths.DATA_ROOT}/mpii/yolov3_detections.pkl')

    examples = []
    with util.BoundedPool(None, 120) as pool:
        for anno_id, (anno, is_train, rect_ids) in enumerate(
                zip(annolist, util.progressbar(s.img_train), s.single_person)):
            if not is_train:
                continue

            image_path = f'{paths.DATA_ROOT}/mpii/images/{anno.image.name}'

            annorect = np.atleast_1d(anno.annorect)
            gt_people = []
            for rect_id, rect in enumerate(annorect):
                if 'annopoints' not in rect or len(rect.annopoints) == 0:
                    continue

                coords = np.full(
                    shape=[joint_info_full.n_joints, 2], fill_value=np.nan, dtype=np.float32)
                for joint in np.atleast_1d(rect.annopoints.point):
                    coords[joint.id] = [joint.x, joint.y]

                bbox = boxlib.expand(boxlib.bb_of_points(coords), 1.25)
                coords = coords[selected_joints]
                ex = Pose2DExample(image_path, coords, bbox=bbox)
                gt_people.append(ex)

            if not gt_people:
                continue

            image_relpath = os.path.relpath(f'images/{anno.image.name}')
            boxes = [box for box in all_boxes[image_relpath] if box[-1] > 0.5]
            if not boxes:
                continue

            iou_matrix = np.array([[boxlib.iou(gt_person.bbox, box[:4])
                                    for box in boxes]
                                   for gt_person in gt_people])
            gt_indices, box_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)

            for i_gt, i_det in zip(gt_indices, box_indices):
                if iou_matrix[i_gt, i_det] > 0.1:
                    ex = gt_people[i_gt]
                    ex.bbox = np.array(boxes[i_det][:4])
                    new_im_path = image_path.replace('mpii', 'mpii_downscaled_yolo')
                    without_ext, ext = os.path.splitext(new_im_path)
                    new_im_path = f'{without_ext}_{rect_id:02d}{ext}'
                    pool.apply_async(make_efficient_example, (ex, new_im_path),
                                     callback=examples.append)

    examples.sort(key=lambda ex: ex.image_path)

    def n_valid_joints(example):
        return np.count_nonzero(np.all(~np.isnan(example.coords), axis=-1))

    examples = [ex for ex in examples if n_valid_joints(ex) > 6]

    return Pose2DDataset(joint_info_used, examples)


@functools.lru_cache()
def get_dataset(dataset_name, *args, **kwargs):
    from options import FLAGS
    logging.debug(f'Making dataset {dataset_name}...')

    def string_to_intlist(string):
        return tuple(int(s) for s in string.split(','))

    kwargs = {**kwargs}
    for subj_key in ['train_subjects', 'valid_subjects', 'test_subjects']:
        if getattr(FLAGS, subj_key):
            kwargs[subj_key] = string_to_intlist(getattr(FLAGS, subj_key))

    return globals()[f'make_{dataset_name}'](*args, **kwargs)
