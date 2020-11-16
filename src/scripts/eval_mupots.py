#!/usr/bin/env python3

import argparse
import glob
import itertools
import re

import numpy as np

import cameralib
import matlabfile
import options
import paths
import util
from options import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, default=None)
    parser.add_argument('--seeds', type=int, default=1)
    options.initialize(parser)
    FLAGS.pred_path = util.ensure_absolute_path(FLAGS.pred_path, f'{paths.DATA_ROOT}/experiments')

    all_true2d, all_true3d, all_true3d_univ = get_all_gt_poses()
    output = evaluate(FLAGS.pred_path, all_true2d, all_true3d, all_true3d_univ)

    print('Root-relative PCK for detected poses (normalized skeletons, bone rescaling) (Table 11)')
    print(to_latex(output['univ', 'rootrel', 'rescale']['pck_matched']))
    print()
    print('Absolute PCK for detected poses (normalized skeletons, bone rescaling) (Table 11)')
    print(to_latex(output['univ', 'nonrootrel', 'rescale']['pck_matched']))
    print(to_latex(output['univ', 'nonrootrel', 'rescale']['auc_matched']))
    print()
    print('Detected poses (unnormalized skeletons, no bone rescaling) (Table 8)')
    numbers = [
        output['nonuniv', 'nonrootrel', 'norescale']['mpjpe14'][-1],
        output['nonuniv', 'rootrel', 'norescale']['mpjpe14'][-1],
        output['nonuniv', 'nonrootrel', 'norescale']['pck_matched'][-1],
        output['nonuniv', 'rootrel', 'norescale']['pck_matched'][-1],
        output['recall'] * 100]
    print(to_latex(numbers))


def evaluate(pred_path, all_true2d, all_true3d, all_true3d_univ, ):
    all_pred2d, all_pred3d = get_all_pred_poses(pred_path)
    (matched_pred3d, matched_true3d, matched_true3d_univ, n_annotated_people, n_matched_people
     ) = match_all_poses(all_pred2d, all_pred3d, all_true2d, all_true3d, all_true3d_univ)
    output = {}
    for univ, rootrel, rescaling in itertools.product(
            ['univ', 'nonuniv'], ['rootrel', 'nonrootrel'], ['rescale', 'norescale']):
        matched_poses_true_now = matched_true3d_univ if univ == 'univ' else matched_true3d
        if rescaling == 'rescale':
            matched_poses_pred_now = rescale_bones(matched_pred3d, matched_poses_true_now)
        else:
            matched_poses_pred_now = matched_pred3d

        diff = matched_poses_pred_now - matched_poses_true_now
        if rootrel == 'rootrel':
            diff -= diff[:, 14:15]
        error = np.linalg.norm(diff, axis=-1)

        metric_names = ['mpjpe14', 'mpjpe16', 'mpjpe17', 'pck_matched', 'auc_matched',
                        'pck_all_anno', 'auc_all_anno']
        mpjpe14 = get_per_sequence_means(error[:, :14], n_matched_people)
        mpjpe16 = get_per_sequence_means(error[:, [*range(14), 15, 16]], n_matched_people)
        mpjpe17 = get_per_sequence_means(error, n_matched_people)
        pck_matched = get_per_sequence_means(error[:, :14] < 150, n_matched_people) * 100
        auc_matched = get_per_sequence_means(
            np.maximum(0, 1 - np.floor(error[:, :14] / 150 * 30 + 1) / 31), n_matched_people) * 100
        pck_all_anno = pck_matched * n_matched_people / n_annotated_people
        auc_all_anno = auc_matched * n_matched_people / n_annotated_people
        metrics = np.array([
            mpjpe14, mpjpe16, mpjpe17, pck_matched, auc_matched, pck_all_anno, auc_all_anno])
        mean_metrics = np.mean(metrics, axis=1, keepdims=True)
        metrics = np.concatenate([metrics, mean_metrics], axis=1)
        output[univ, rootrel, rescaling] = dict(zip(metric_names, metrics))
    recall = np.sum(n_matched_people) / np.sum(n_annotated_people)
    output['recall'] = recall
    return output


def to_latex(numbers):
    return ' & '.join([f'{x:.1f}' for x in numbers])


def get_per_sequence_means(metric, n_matched_people):
    metric = np.mean(metric, axis=1)  # averaging over the joints
    per_sequence_chunks = np.split(metric, np.cumsum(n_matched_people[:-1]))
    return np.array([np.mean(chunk, axis=0) for chunk in per_sequence_chunks])


def get_all_gt_poses():
    n_seq = 20
    all_true2d = [[] for _ in range(n_seq)]
    all_true3d = [[] for _ in range(n_seq)]
    all_true3d_univ = [[] for _ in range(n_seq)]
    for i_seq in range(n_seq):
        anno_path = f'{paths.DATA_ROOT}/mupots/TS{i_seq + 1}/annot.mat'
        annotations = matlabfile.load(anno_path).annotations
        n_frames, n_people = annotations.shape
        for i_frame in range(len(annotations)):
            valid_annotations = [
                annotations[i_frame, i_person] for i_person in range(n_people)
                if annotations[i_frame, i_person].isValidFrame]
            all_true2d[i_seq].append(np.array([anno.annot2.T for anno in valid_annotations]))
            all_true3d[i_seq].append(np.array([anno.annot3.T for anno in valid_annotations]))
            all_true3d_univ[i_seq].append(
                np.array([anno.univ_annot3.T for anno in valid_annotations]))

    return all_true2d, all_true3d, all_true3d_univ


def get_all_pred_poses(path):
    results = np.load(path, allow_pickle=True)
    joint_remapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 14, 15]
    n_seq = 20
    n_frames_per_seq = [
        len(glob.glob(f'{paths.DATA_ROOT}/mupots/TS{i_seq + 1}/img_*.jpg'))
        for i_seq in range(n_seq)]

    all_pred2d = [[[] for _ in range(n_frames)] for n_frames in n_frames_per_seq]
    all_pred3d = [[[] for _ in range(n_frames)] for n_frames in n_frames_per_seq]
    intrinsic_matrices = util.load_json(f'{paths.DATA_ROOT}/mupots/camera_intrinsics.json')

    for image_path, coords3d_pred in zip(results['image_path'], results['coords3d_pred']):
        m = re.match(r'.+/TS(?P<i_seq>\d+)/img_(?P<i_frame>\d+)\.jpg', image_path.decode('utf8'))
        i_seq = int(m['i_seq']) - 1
        i_frame = int(m['i_frame'])
        coords3d_pred = coords3d_pred[joint_remapping]
        camera = cameralib.Camera(
            intrinsic_matrix=intrinsic_matrices[f'TS{i_seq + 1}'], world_up=(0, -1, 0))
        coords2d = camera.world_to_image(coords3d_pred)
        all_pred2d[i_seq][i_frame].append(coords2d)
        all_pred3d[i_seq][i_frame].append(coords3d_pred)

    all_pred2d = [[np.array(frame_preds) for frame_preds in seq_preds] for seq_preds in all_pred2d]
    all_pred3d = [[np.array(frame_preds) for frame_preds in seq_preds] for seq_preds in all_pred3d]

    return all_pred2d, all_pred3d


def match_all_poses(all_pred2d, all_pred3d, all_true2d, all_true3d, all_true3d_univ):
    n_seq = len(all_pred2d)
    n_annotated_people = np.zeros(n_seq, dtype=int)
    n_matched_people = np.zeros(n_seq, dtype=int)
    matched_poses_true = []
    matched_poses_true_univ = []
    matched_poses_pred = []
    for i_seq, seq_poses in enumerate(
            zip(all_true2d, all_true3d, all_true3d_univ, all_pred2d, all_pred3d)):
        for i_frame, (true2d, true3d, true3d_univ, pred2d, pred3d) in enumerate(zip(*seq_poses)):
            matching = match_poses(true2d[:, 1:14], pred2d[:, 1:14])
            for i_true, i_pred in matching:
                matched_poses_true.append(true3d[i_true])
                matched_poses_true_univ.append(true3d_univ[i_true])
                matched_poses_pred.append(pred3d[i_pred])

            n_annotated_people[i_seq] += len(true2d)
            n_matched_people[i_seq] += len(matching)

    matched_poses_true = np.asarray(matched_poses_true)
    matched_poses_true_univ = np.asarray(matched_poses_true_univ)
    matched_poses_pred = np.asarray(matched_poses_pred)
    return (matched_poses_pred, matched_poses_true, matched_poses_true_univ, n_annotated_people,
            n_matched_people)


def match_poses(poses2d_true, poses2d_pred, thresh=40):
    matching = []
    for i_true, pose2d_true in enumerate(poses2d_true):
        best_n_matching_joints = 0
        best_i_pred = None

        for i_pred, pose2d_pred in enumerate(poses2d_pred):
            if any(match[1] == i_pred for match in matching):  # Already matched prediction
                continue

            is_error_small = np.max(np.abs(pose2d_true - pose2d_pred), axis=1) < thresh
            valid_pred = np.any(pose2d_pred != 0, axis=-1)
            n_matching_joints = np.count_nonzero(np.logical_and(is_error_small, valid_pred))

            if n_matching_joints > best_n_matching_joints:
                best_i_pred = i_pred
                best_n_matching_joints = n_matching_joints

        if best_i_pred is not None:
            matching.append((i_true, best_i_pred))

    return matching


def rescale_bones(pred, gt):
    bones_in_safe_traversal_order = [
        (15, 14), (1, 15), (0, 1), (16, 1), (2, 1), (3, 2), (4, 3), (5, 1),
        (6, 5), (7, 6), (8, 14), (9, 8), (10, 9), (11, 14), (12, 11), (13, 12)]
    mapped_pose = pred.copy()
    for i_joint, i_parent in bones_in_safe_traversal_order:
        gt_bone_length = np.linalg.norm(gt[:, i_joint] - gt[:, i_parent], axis=-1, keepdims=True)
        pred_bone_vector = pred[:, i_joint] - pred[:, i_parent]
        pred_bone_length = np.linalg.norm(pred_bone_vector, axis=-1, keepdims=True)
        rescaled_pred_bone_vector = pred_bone_vector * gt_bone_length / pred_bone_length
        mapped_pose[:, i_joint] = mapped_pose[:, i_parent] + rescaled_pred_bone_vector
    return mapped_pose


if __name__ == '__main__':
    main()
