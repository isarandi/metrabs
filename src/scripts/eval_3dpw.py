#!/usr/bin/env python
import argparse
import glob
import os
import pickle

import numpy as np
import scipy.ndimage

import cameralib
import options
import paths
import util
import util3d
from options import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, default=None)
    parser.add_argument('--procrustes', action=options.YesNoAction)
    parser.add_argument('--acausal-smoothing', action=options.YesNoAction)
    parser.add_argument('--causal-smoothing', action=options.YesNoAction)
    options.initialize(parser)
    FLAGS.pred_path = util.ensure_absolute_path(FLAGS.pred_path, f'{paths.DATA_ROOT}/experiments')

    poses3d_true_dict = get_all_gt_poses()
    poses3d_pred_dict = get_all_pred_poses()

    all_pred3d = np.array([poses3d_pred_dict[relpath] for relpath in poses3d_true_dict])
    all_true3d = np.array(list(poses3d_true_dict.values()))
    all_pred3d -= all_pred3d[:, :1]
    all_true3d -= all_true3d[:, :1]
    all_pred3d_aligned = util3d.rigid_align_many(all_pred3d, all_true3d, scale_align='rigid+scale')

    dist = np.linalg.norm(all_true3d - all_pred3d, axis=-1)
    dist_aligned = np.linalg.norm(all_true3d - all_pred3d_aligned, axis=-1)

    mpjpe = np.mean(dist)
    mpjpe_pa = np.mean(dist_aligned)
    major_dist = dist[:, [1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21]]
    pck = np.mean(major_dist / 50 <= 1) * 100
    auc = np.mean(np.maximum(0, 1 - (np.floor(major_dist / 199 * 50) + 0.5) / 50)) * 100
    print('MPJPE & MPJPE_PA & PCK & AUC')
    print(to_latex([mpjpe, mpjpe_pa, pck, auc]))


def to_latex(numbers):
    return ' & '.join([f'{x:.5f}' for x in numbers])


def get_all_gt_poses():
    all_valid_poses = {}
    seq_filepaths = glob.glob(f'{paths.DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl')
    for filepath in seq_filepaths:
        with open(filepath, 'rb') as f:
            seq = pickle.load(f, encoding='latin1')
        seq_name = seq['sequence']
        intrinsics = seq['cam_intrinsics']
        extrinsics_per_frame = seq['cam_poses']

        for i_person, (coord3d_seq, coords2d_seq, trans_seq, camvalid_seq) in enumerate(zip(
                seq['jointPositions'], seq['poses2d'], seq['trans'], seq['campose_valid'])):
            for i_frame, (coords3d, coords2d, trans, extrinsics, campose_valid) in enumerate(
                    zip(coord3d_seq, coords2d_seq, trans_seq, extrinsics_per_frame, camvalid_seq)):
                if not campose_valid or np.all(coords2d == 0):
                    continue
                image_relpath = f'imageFiles/{seq_name}/image_{i_frame:05d}.jpg'
                camera = cameralib.Camera(
                    extrinsic_matrix=extrinsics, intrinsic_matrix=intrinsics,
                    world_up=(0, 1, 0))
                camera.t *= 1000
                world_coords = coords3d.reshape(-1, 3) * 1000
                camcoords = camera.world_to_camera(world_coords)
                all_valid_poses[image_relpath, i_person] = camcoords

    return all_valid_poses


def get_all_pred_poses():
    pred_filepaths = glob.glob(f'{FLAGS.pred_path}/**/*.pkl', recursive=True)
    all_pred_poses = {}
    for filepath in pred_filepaths:
        seq_name = os.path.splitext(os.path.basename(filepath))[0]
        preds = util.load_pickle(filepath)['jointPositions']
        if FLAGS.causal_smoothing:
            preds = causal_smooth(preds)
        elif FLAGS.acausal_smoothing:
            preds = acausal_smooth(preds)
        for i_person, person_preds in enumerate(preds):
            for i_frame, pred in enumerate(person_preds):
                image_relpath = f'imageFiles/{seq_name}/image_{i_frame:05d}.jpg'
                all_pred_poses[image_relpath, i_person] = pred * 1000

    return all_pred_poses


def causal_smooth(tracks):
    kernel = np.array([6, 2, 1], np.float32)
    kernel /= np.sum(kernel)
    return scipy.ndimage.convolve1d(tracks, kernel, axis=1, origin=-1, mode='reflect')


def acausal_smooth(tracks):
    kernel = np.array([1, 2, 6, 2, 1], np.float32)
    kernel /= np.sum(kernel)
    return scipy.ndimage.convolve1d(tracks, kernel, axis=1, mode='reflect')


if __name__ == "__main__":
    main()
