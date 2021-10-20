#!/usr/bin/env python3
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import itertools
import re

import numpy as np
import spacepy

import data.h36m
import options
import paths
import util
from options import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, default=None)
    parser.add_argument('--procrustes', action=options.BoolAction)
    parser.add_argument('--only-S11', action=options.BoolAction)
    parser.add_argument('--seeds', type=int, default=1)

    # The root joint is the last if this is set, else the first
    parser.add_argument('--root-last', action=options.BoolAction)
    options.initialize(parser)
    FLAGS.pred_path = util.ensure_absolute_path(FLAGS.pred_path, f'{paths.DATA_ROOT}/experiments')

    all_image_relpaths, all_true3d = get_all_gt_poses()
    activities = np.array([re.search(f'Images/(.+?)\.', path)[1].split(' ')[0]
                           for path in all_image_relpaths])

    if FLAGS.seeds > 1:
        mean_per_seed, std_per_seed = evaluate_multiple_seeds(all_true3d, activities)
        print(to_latex(mean_per_seed))
        print(to_latex(std_per_seed))
    else:
        metrics = evaluate(FLAGS.pred_path, all_true3d, activities)
        print(to_latex(metrics))


def evaluate_multiple_seeds(all_true3d, activities):
    seed_pred_paths = [FLAGS.pred_path.replace('seed1', f'seed{i + 1}') for i in range(FLAGS.seeds)]
    metrics_per_seed = np.array([evaluate(p, all_true3d, activities) for p in seed_pred_paths])
    mean_per_seed = np.mean(metrics_per_seed, axis=0)
    std_per_seed = np.std(metrics_per_seed, axis=0)
    return mean_per_seed, std_per_seed


def evaluate(pred_path, all_true3d, activities):
    all_pred3d = get_all_pred_poses(pred_path)
    if len(all_pred3d) != len(all_true3d):
        raise Exception(f'Unequal sample count! Pred: {len(all_pred3d)}, GT: {len(all_true3d)}')

    i_root = -1 if FLAGS.root_last else 0
    all_pred3d -= all_pred3d[:, i_root, np.newaxis]
    all_true3d -= all_true3d[:, i_root, np.newaxis]

    ordered_activities = (
            'Directions Discussion Eating Greeting Phoning Posing Purchases ' +
            'Sitting SittingDown Smoking Photo Waiting Walking WalkDog WalkTogether').split()
    if FLAGS.procrustes:
        all_pred3d = tfu3d.rigid_align(all_pred3d, all_true3d, scale_align=True)
    dist = np.linalg.norm(all_true3d - all_pred3d, axis=-1)
    overall_mean_error = np.mean(dist)
    metrics = [np.mean(dist[activities == activity]) for activity in ordered_activities]
    metrics.append(overall_mean_error)
    return metrics


def to_latex(numbers):
    return ' & '.join([f'{x:.1f}' for x in numbers])


def load_coords(path):
    if FLAGS.root_last:
        i_relevant_joints = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27, 0]
    else:
        i_relevant_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
    with spacepy.pycdf.CDF(path) as cdf_file:
        coords_raw = np.array(cdf_file['Pose'], np.float32)[0]
    coords_new_shape = [coords_raw.shape[0], -1, 3]
    return coords_raw.reshape(coords_new_shape)[:, i_relevant_joints]


def get_all_gt_poses():
    camera_names = ['54138969', '55011271', '58860488', '60457274']
    frame_step = 64
    all_world_coords = []
    all_image_relpaths = []
    for i_subj in [9, 11]:
        for activity, cam_id in itertools.product(data.h36m.get_activity_names(i_subj), range(4)):
            # Corrupt data in original release:
            # if i_subj == 11 and activity == 'Directions' and cam_id == 0:
            #    continue
            camera_name = camera_names[cam_id]
            pose_folder = f'{paths.DATA_ROOT}/h36m/S{i_subj}/MyPoseFeatures'
            coord_path = f'{pose_folder}/D3_Positions/{activity}.cdf'
            world_coords = load_coords(coord_path)
            n_frames_total = len(world_coords)
            world_coords = world_coords[::frame_step]
            all_world_coords.append(world_coords)
            image_relfolder = f'h36m/S{i_subj}/Images/{activity}.{camera_name}'
            all_image_relpaths += [
                f'{image_relfolder}/frame_{i_frame:06d}.jpg'
                for i_frame in range(0, n_frames_total, frame_step)]

    order = np.argsort(all_image_relpaths)
    all_world_coords = np.concatenate(all_world_coords, axis=0)[order]
    all_image_relpaths = np.array(all_image_relpaths)[order]
    if FLAGS.only_S11:
        needed = ['S11' in p for p in all_image_relpaths]
        return all_image_relpaths[needed], all_world_coords[needed]
    return all_image_relpaths, all_world_coords


def get_all_pred_poses(path):
    results = np.load(path, allow_pickle=True)
    order = np.argsort(results['image_path'])
    image_paths = results['image_path'][order]
    if FLAGS.only_S11:
        needed = ['S11' in p for p in image_paths]
        return results['coords3d_pred_world'][order][needed]

    return results['coords3d_pred_world'][order]


if __name__ == '__main__':
    main()
