#!/usr/bin/env python3
import argparse
import re

import h5py
import numpy as np

import data.mpi_inf_3dhp
import options
import paths
import util
from options import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, default=None)
    parser.add_argument('--universal-skeleton', action=options.BoolAction)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--corrected-TS6', action=options.BoolAction, default=True)
    parser.add_argument('--root-last', action=options.BoolAction, default=False)
    options.initialize(parser)
    FLAGS.pred_path = util.ensure_absolute_path(FLAGS.pred_path, f'{paths.DATA_ROOT}/experiments')

    all_image_relpaths, all_true3d, activities = get_all_gt_poses()

    def get_scene_name(image_path):
        i_subject = int(re.search(r'/TS(\d+?)/', image_path)[1])
        return ['green-screen', 'no-green-screen', 'outdoor'][(i_subject - 1) // 2]

    scene_names = np.array([get_scene_name(path) for path in all_image_relpaths])
    if FLAGS.seeds > 1:
        mean_per_seed, std_per_seed = evaluate_multiple_seeds(all_true3d, activities, scene_names)
        print(to_latex(mean_per_seed))
        print(to_latex(std_per_seed))
    else:
        metrics = evaluate(FLAGS.pred_path, all_true3d, activities, scene_names)
        print(to_latex(metrics))


def evaluate_multiple_seeds(all_true3d, activity_names, scene_names):
    seed_pred_paths = [FLAGS.pred_path.replace('seed1', f'seed{i + 1}') for i in range(5)]
    metrics_per_seed = np.array([
        evaluate(p, all_true3d, activity_names, scene_names) for p in seed_pred_paths])
    mean_per_seed = np.mean(metrics_per_seed, axis=0)
    std_per_seed = np.std(metrics_per_seed, axis=0)
    return mean_per_seed, std_per_seed


def evaluate(pred_path, all_true3d, activity_names, scene_names):
    all_pred3d = get_all_pred_poses(pred_path)
    assert len(all_pred3d) == len(all_true3d)

    # Make it root relative
    # Joint order is:
    # 'htop,neck,rsho,relb,rwri,lsho,lelb,lwri,rhip,rkne,rank,lhip,lkne,lank,pelv,spin,head'
    i_root = 14
    all_pred3d -= all_pred3d[:, i_root, np.newaxis]
    all_true3d -= all_true3d[:, i_root, np.newaxis]
    dist = np.linalg.norm(all_true3d - all_pred3d, axis=-1)
    overall_mean_error = np.mean(dist)
    ordered_actions = (
        'Stand/Walk,Exercise,Sit on Chair,Reach/Crouch,On Floor,Sports,Misc.'.split(','))
    ordered_scenes = ['green-screen', 'no-green-screen', 'outdoor']

    # The PCK and AUC values are computed on a 14-joint subset
    rel_dist = (dist / 150)[:, :14]
    pck_per_activity = [
        get_pck(rel_dist[activity_names == activity]) for activity in ordered_actions]
    pck_per_scene = [get_pck(rel_dist[scene_names == scene]) for scene in ordered_scenes]
    return np.array([
        *pck_per_activity, *pck_per_scene, get_pck(rel_dist), get_auc(rel_dist),
        overall_mean_error])


def get_pck(rel_dists):
    return np.mean(rel_dists < 1) * 100


def get_auc(rel_dists):
    return np.mean(np.maximum(0, 1 - np.floor(rel_dists * 30 + 1) / 31)) * 100


def to_latex(numbers):
    return ' & '.join([f'{x:.1f}' for x in numbers])


def get_all_gt_poses():
    activity_names = [
        'Stand/Walk', 'Exercise', 'Sit on Chair', 'Reach/Crouch', 'On Floor', 'Sports', 'Misc.']
    images_relpaths = []
    world_poses = []
    activities = []
    cam1_4 = data.mpi_inf_3dhp.get_test_camera_subj1_4()
    cam5_6 = data.mpi_inf_3dhp.get_test_camera_subj5_6()
    anno_key = 'univ_annot3' if FLAGS.universal_skeleton else 'annot3'

    for i_subject in range(1, 7):
        with h5py.File(f'{paths.DATA_ROOT}/3dhp/TS{i_subject}/annot_data.mat', 'r') as m:
            cam3d_coords = np.array(m[anno_key])[:, 0]
            valid_frames = np.where(m['valid_frame'][:, 0])[0]
            activity_ids = m['activity_annotation'][:, 0].astype(int) - 1

        camera = cam1_4 if i_subject <= 4 else cam5_6
        for i_frame in valid_frames:
            images_relpaths.append(f'3dhp/TS{i_subject}/imageSequence/img_{i_frame + 1:06d}.jpg')
            world_poses.append(camera.camera_to_world(cam3d_coords[i_frame]))
            activities.append(activity_names[activity_ids[i_frame]])
    return np.array(images_relpaths), np.array(world_poses), np.array(activities)


def get_all_pred_poses(path):
    results = np.load(path, allow_pickle=True)
    order = np.argsort(results['image_path'])
    pred = results['coords3d_pred_world']
    return pred[order]


if __name__ == '__main__':
    main()
