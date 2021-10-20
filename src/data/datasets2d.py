import functools
from options import logger
import os
import os.path

import numpy as np
import pycocotools.coco
import scipy.optimize

import boxlib
import data.joint_filtering
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
    f'{paths.CACHE_DIR}/mpii_yolo.pkl', min_time="2021-06-01T21:39:35")
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
        for anno_id, (anno, is_train) in enumerate(
                zip(annolist, util.progressbar(s.img_train))):
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
                    new_im_path = f'{without_ext}_{i_gt:02d}{ext}'
                    pool.apply_async(make_efficient_example, (ex, new_im_path),
                                     callback=examples.append)

    examples.sort(key=lambda ex: ex.image_path)

    def n_valid_joints(example):
        return np.count_nonzero(np.all(~np.isnan(example.coords), axis=-1))

    examples = [ex for ex in examples if n_valid_joints(ex) > 6]

    return Pose2DDataset(joint_info_used, examples)


def make_coco_reduced(single_person=False, face=True):
    joint_names = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,lelb,lwri'
    if face:
        joint_names += ',nose,leye,reye,lear,rear'

    edges = 'lelb-lwri,relb-rwri,lhip-lkne-lank,rhip-rkne-rank'
    joint_info = JointInfo(joint_names, edges)
    ds = data.joint_filtering.convert_dataset(make_coco(single_person), joint_info)

    body_joint_names = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,lelb,lwri'.split(',')
    body_joint_ids = [joint_info.ids[name] for name in body_joint_names]

    def n_valid_body_joints(example):
        return np.count_nonzero(
            np.all(~np.isnan(example.coords[body_joint_ids]), axis=-1))

    ds.examples[TRAIN] = [ex for ex in ds.examples[TRAIN] if n_valid_body_joints(ex) > 6]
    return ds


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/cached_coco.dat', min_time="2020-02-01T02:53:21")
def make_coco(single_person=True):
    joint_info = JointInfo(
        'nose,leye,reye,lear,rear,lsho,rsho,lelb,relb,lwri,rwri,lhip,rhip,lkne,rkne,lank,rank',
        'lsho-lelb-lwri,rsho-relb-rwri,lhip-lkne-lank,rhip-rkne-rank,lear-leye-nose-reye-rear')
    n_joints = joint_info.n_joints
    learning_phase_shortnames = {TRAIN: 'train', VALID: 'val', TEST: 'test'}
    UNLABELED = 0
    OCCLUDED = 1
    VISIBLE = 2
    iou_threshold = 0.1 if single_person else 0.5

    suffix = '' if single_person else '_multi'
    examples_per_phase = {TRAIN: [], VALID: []}
    with util.BoundedPool(None, 120) as pool:
        for example_phase in (TRAIN, VALID):
            phase_shortname = learning_phase_shortnames[example_phase]
            coco_filepath = (
                f'{paths.DATA_ROOT}/coco/annotations/person_keypoints_{phase_shortname}2014.json')
            coco = pycocotools.coco.COCO(coco_filepath)

            impath_to_examples = {}
            for ann in coco.anns.values():
                filename = coco.imgs[ann['image_id']]['file_name']
                image_path = f'{paths.DATA_ROOT}/coco/{phase_shortname}2014/{filename}'

                joints = np.array(ann['keypoints']).reshape([-1, 3])
                visibilities = joints[:, 2]
                coords = joints[:, :2].astype(np.float32).copy()
                n_visible_joints = np.count_nonzero(visibilities == VISIBLE)
                n_occluded_joints = np.count_nonzero(visibilities == OCCLUDED)
                n_labeled_joints = n_occluded_joints + n_visible_joints

                if n_visible_joints >= n_joints / 3 and n_labeled_joints >= n_joints / 2:
                    coords[visibilities == UNLABELED] = np.nan
                    bbox_pt1 = np.array(ann['bbox'][0:2], np.float32)
                    bbox_wh = np.array(ann['bbox'][2:4], np.float32)
                    bbox = np.array([*bbox_pt1, *bbox_wh])
                    ex = Pose2DExample(image_path, coords, bbox=bbox)
                    impath_to_examples.setdefault(image_path, []).append(ex)

            n_images = len(impath_to_examples)
            for impath, examples in util.progressbar(impath_to_examples.items(), total=n_images):
                for i_example, example in enumerate(examples):
                    box = boxlib.expand(boxlib.bb_of_points(example.coords), 1.25)
                    if np.max(box[2:]) < 200:
                        continue

                    if single_person:
                        other_boxes = [boxlib.expand(boxlib.bb_of_points(e.coords), 1.25)
                                       for e in examples if e is not example]
                        ious = np.array([boxlib.iou(b, box) for b in other_boxes])
                        usable = np.all(ious < iou_threshold)
                    else:
                        usable = True

                    if usable:
                        new_im_path = impath.replace('coco', 'coco_downscaled' + suffix)
                        without_ext, ext = os.path.splitext(new_im_path)
                        new_im_path = f'{without_ext}_{i_example:02d}{ext}'
                        pool.apply_async(
                            make_efficient_example, (example, new_im_path),
                            callback=examples_per_phase[example_phase].append)

    examples_per_phase[TRAIN].sort(key=lambda ex: ex.image_path)
    examples_per_phase[VALID].sort(key=lambda ex: ex.image_path)
    return Pose2DDataset(joint_info, examples_per_phase[TRAIN], examples_per_phase[VALID])


@functools.lru_cache()
def get_dataset(dataset_name, *args, **kwargs):
    logger.debug(f'Making dataset {dataset_name}...')
    return globals()[f'make_{dataset_name}'](*args, **kwargs)
