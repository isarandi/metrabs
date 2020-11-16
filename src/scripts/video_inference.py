#!/usr/bin/env python3
import argparse
import glob
import os
import pickle
import queue
import threading

import imageio
import numpy as np
import scipy.ndimage
import scipy.optimize
import tensorflow as tf

import boxlib
import cameralib
import data.datasets3d
import improc
import options
import paths
import util
from options import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--darknet-dir', type=str)
    parser.add_argument('--gt-assoc', action=options.YesNoAction)
    parser.add_argument('--precomputed-detections', action=options.YesNoAction)
    parser.add_argument('--batched', action=options.YesNoAction)
    parser.add_argument('--crops', type=int, default=5)
    parser.add_argument('--detector-flip-aug', action=options.YesNoAction)
    parser.add_argument('--detector-path', type=str)
    parser.add_argument('--antialias', action=options.YesNoAction)
    parser.add_argument('--real-intrinsics', action=options.YesNoAction)
    parser.add_argument('--causal-smoothing', action=options.YesNoAction)
    parser.add_argument('--gui', action=options.YesNoAction)
    options.initialize(parser)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    detector = tf.saved_model.load(FLAGS.detector_path)
    pose_estimator = tf.saved_model.load(FLAGS.model_path)

    joint_names = [b.decode('utf8') for b in pose_estimator.crop_model.joint_names.numpy()]
    edges = pose_estimator.crop_model.joint_edges.numpy()
    ji3d = data.datasets3d.JointInfo(joint_names, edges)

    if FLAGS.gui:
        q = queue.Queue(30)
        visualizer_thread = threading.Thread(target=main_visualize, args=(q, ji))
        visualizer_thread.start()
    else:
        q = None

    seq_filepaths = sorted(glob.glob(f'{paths.DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl'))
    seq_filepaths = [x for x in seq_filepaths if 'capoeira' in x]
    seq_names = [os.path.basename(p).split('.')[0] for p in seq_filepaths]
    subdir = 'gtassoc' if FLAGS.gt_assoc else 'nogtassoc'
    subdirpath = f'{FLAGS.output_dir}/{subdir}'

    for seq_name, seq_filepath in util.progressbar(zip(seq_names, seq_filepaths)):
        already_done_files = glob.glob(f'{subdirpath}/*/*.pkl')
        if any(seq_name in p for p in already_done_files):
            continue
        print(seq_name)
        frame_paths = sorted(
            glob.glob(f'{paths.DATA_ROOT}/3dpw/imageFiles/{seq_name}/image_*.jpg'))
        poses2d_true = get_poses2d_3dpw(seq_name)
        camera = get_3dpw_camera(seq_filepath) if FLAGS.real_intrinsics else None
        tracks = track_them(
            detector, pose_estimator, frame_paths, poses2d_true, ji2d, ji3d, q, camera=camera)
        save_result_file(seq_name, subdirpath, tracks)


def track_them(
        detector, pose_estimator, frame_paths, poses2d_true, joint_info2d, joint_info3d, q,
        n_tracks=None, camera=None):
    if poses2d_true is not None:
        n_tracks = poses2d_true.shape[1]
        prev_poses2d_pred_ordered = np.zeros((n_tracks, joint_info3d.n_joints, 2))
        tracks = [[] for _ in range(n_tracks)]
    elif n_tracks is not None:
        prev_poses2d_pred_ordered = None
        tracks = [[(-1, np.full((joint_info3d.n_joints, 3), fill_value=np.inf))]
                  for _ in range(n_tracks)]
    else:
        prev_poses2d_pred_ordered = None
        tracks = []

    dataset = tf.data.Dataset.from_tensor_slices(frame_paths)
    dataset = dataset.map(load_image, tf.data.experimental.AUTOTUNE, deterministic=False)

    if FLAGS.batched:
        dataset = predict_in_batches(dataset, camera, detector, pose_estimator)

    for i_frame, item in enumerate(util.progressbar(dataset)):
        if FLAGS.batched:
            frame, detections, poses = item
            crop_boxes = detections
            if camera is None:
                camera = get_main_camera(frame.shape)
        else:
            frame = item[0].numpy()
            if camera is None:
                camera = get_main_camera(frame.shape)
            detections = detector(frame[np.newaxis], 0.5, 0.4)[0].numpy()

            # Inject new boxes based on the previous poses
            crop_boxes = get_crop_boxes(i_frame, camera, tracks, detections)
            poses = pose_estimator.predict_single_image(
                frame, camera.intrinsic_matrix, crop_boxes[..., :4], 65, FLAGS.crops).numpy()

        pose_sanity = [is_pose_sane(pose, mean_bone_lengths, ji) for pose in poses]
        poses = poses[pose_sanity]
        confs = np.array(crop_boxes)[:, 4][pose_sanity]
        poses, confs = nms_pose(poses, confs)

        if FLAGS.gt_assoc or (i_frame == 0 and poses2d_true is not None):
            poses2d_pred = [camera.camera_to_image(pose) for pose in poses]
            poses_ordered, prev_poses2d_pred_ordered = associate_predictions(
                poses, poses2d_pred, poses2d_true[i_frame], prev_poses2d_pred_ordered,
                joint_info3d, joint_info2d)
            for pose, track in zip(poses_ordered, tracks):
                if not np.any(np.isnan(pose)):
                    track.append((i_frame, pose))
        else:
            update_tracks(i_frame, tracks, poses, confs)

        poses = np.array([t[-1][1] for t in tracks if t])
        if q is not None:
            for box in detections:
                improc.draw_box(frame, box, color=(255, 0, 0), thickness=5)
            q.put((frame, poses, camera))

    return tracks


@tf.function
def load_image(data):
    return (tf.image.decode_jpeg(
        tf.io.read_file(data), fancy_upscaling=False, dct_method='INTEGER_FAST'),)


def predict_in_batches(dataset, camera, detector, pose_estimator):
    for (frame_batch,) in dataset.batch(32):
        if camera is None:
            imshape = tf.shape(frame_batch)[1:3].numpy()
            intrinsics = get_main_camera(imshape).intrinsic_matrix[np.newaxis]
        else:
            intrinsics = camera.intrinsic_matrix[np.newaxis]
        detections = detector(frame_batch, 0.5, 0.4)
        poses = pose_estimator.predict_multi_image(
            frame_batch, intrinsics, detections[..., :4], 65, FLAGS.crops)
        yield from zip(frame_batch.numpy(), detections.numpy(), poses.numpy())


def get_3dpw_camera(seq_filepath):
    with open(seq_filepath, 'rb') as f:
        intr = pickle.load(f, encoding='latin1')['cam_intrinsics']
        return cameralib.Camera(intrinsic_matrix=intr, world_up=[0, -1, 0])


def get_poses2d_3dpw(seq_name):
    seq_filepaths = glob.glob(f'{paths.DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl')
    filepath = next(p for p in seq_filepaths if os.path.basename(p) == f'{seq_name}.pkl')
    with open(filepath, 'rb') as f:
        seq = pickle.load(f, encoding='latin1')
    return np.transpose(np.array(seq['poses2d']), [1, 0, 3, 2])  # [Frame, Track, Joint, Coord]


def pose2d_auc(pose2d_pred, pose2d_true, prev_pose2d_pred, joint_info3d, joint_info2d):
    pose2d_true = pose2d_true.copy()
    pose2d_true[pose2d_true[:, 2] < 0.2] = np.nan
    selected_joints = 'lsho,rsho,lelb,relb,lhip,rhip,lkne,rkne'.split(',')
    indices_true = [joint_info2d.ids[name] for name in selected_joints]
    indices_pred = [joint_info3d.ids[name] for name in selected_joints]
    size = np.linalg.norm(pose2d_pred[joint_info3d.ids.rsho] - pose2d_pred[joint_info3d.ids.lhip])
    dist = np.linalg.norm(pose2d_true[indices_true, :2] - pose2d_pred[indices_pred], axis=-1)
    if np.count_nonzero(~np.isnan(dist)) < 5:
        dist = np.linalg.norm(prev_pose2d_pred[indices_pred] - pose2d_pred[indices_pred], axis=-1)
    return np.nanmean(np.maximum(0, 1 - dist / size))


def get_main_camera(imshape):
    f = np.max(imshape[:2]) / (np.tan(np.deg2rad(60) / 2) * 2)
    intrinsic_matrix = np.array([[f, 0, imshape[1] / 2], [0, f, imshape[0] / 2], [0, 0, 1]])
    return cameralib.Camera(intrinsic_matrix=intrinsic_matrix, world_up=(0, -1, 0))


def is_pose_sane(pose, sane_bone_lengths, ji):
    if np.any(np.isnan(pose)):
        return False

    bone_lengths = np.array(
        [np.linalg.norm(pose[i] - pose[j], axis=-1) for i, j in ji.stick_figure_edges])
    bone_length_relative = bone_lengths / sane_bone_lengths
    bone_length_diff = np.abs(bone_lengths - sane_bone_lengths)

    with np.errstate(invalid='ignore'):
        relsmall = bone_length_relative < 0.1
        relbig = bone_length_relative > 3
        absdiffbig = bone_length_diff > 300

    insane = np.any(np.logical_and(np.logical_or(relbig, relsmall), absdiffbig))
    return not insane


def get_crop_boxes(i_frame, camera, tracks, detections):
    live_tracks = [track for track in tracks if len(track) > 30 and i_frame - track[-1][0] < 10]
    last_live_poses = np.array([track[-1][1] for track in live_tracks])
    shadow_boxes = [boxlib.expand(boxlib.bb_of_points(camera.camera_to_image(p)), 1.2) for p in
                    last_live_poses]

    crop_boxes = list(detections)
    for shadow_box in shadow_boxes:
        for box in crop_boxes:
            if boxlib.iou(box[:4], shadow_box[:4]) > 0.65:
                break
        else:
            crop_boxes.append([*shadow_box, 0])

    if len(crop_boxes) == 0:
        return np.zeros((0, 5), np.float32)
    return np.array(crop_boxes, np.float32)


def auc_for_nms(p1, p2, thresh=1000, topk=3):
    if np.any(~np.isfinite(p1)) or np.any(~np.isfinite(p2)):
        return -1
    rel_dists = np.linalg.norm(p1 - p2, axis=-1) / thresh
    rel_dists = np.sort(rel_dists)[:topk]
    return np.mean(np.maximum(0, 1 - rel_dists))


def update_tracks(i_frame, tracks, current_poses, confs):
    if not tracks:
        tracks += [[(i_frame, p)] for p, c in zip(current_poses, confs) if c > 0]
        return

    prev_poses = [track[-1][1] for track in tracks]
    auc_matrix = np.array(
        [[auc_for_nms(p1, p2, 500, 20) for p1 in current_poses] for p2 in prev_poses])
    prev_indices, current_indices = scipy.optimize.linear_sum_assignment(-auc_matrix)
    used_cadidate_indices = []
    used_track_indices = []
    for pi, ci in zip(prev_indices, current_indices):
        track = tracks[pi]
        if auc_matrix[pi, ci] > 0 or auc_matrix[pi, ci] == -1:
            track.append((i_frame, current_poses[ci]))
            used_cadidate_indices.append(ci)
            used_track_indices.append(pi)

    current_poses = np.array(current_poses)
    confs = np.array(confs)
    unused_candidates = np.array([
        ci not in used_cadidate_indices for ci in range(len(current_poses))])
    unused_poses = iter(current_poses[unused_candidates][np.argsort(-confs[unused_candidates])])

    for pi, track in enumerate(tracks):
        if pi not in used_track_indices and (track and i_frame - track[-1][0] > 30):
            next_best_unused_pose = next(unused_poses, None)
            if next_best_unused_pose is None:
                return
            track.append((i_frame, next_best_unused_pose))


def associate_predictions(
        poses3d_pred, poses2d_pred, poses2d_true, prev_poses2d_pred_ordered,
        joint_info3d, joint_info2d):
    auc_matrix = np.array([
        [pose2d_auc(pose_pred, pose_true, prev_pose, joint_info3d, joint_info2d)
         for pose_pred in poses2d_pred]
        for pose_true, prev_pose in zip(poses2d_true, prev_poses2d_pred_ordered)])

    true_indices, pred_indices = scipy.optimize.linear_sum_assignment(-auc_matrix)
    n_true_poses = len(poses2d_true)

    result = np.full((n_true_poses, joint_info3d.n_joints, 3), np.nan)
    poses2d_pred_ordered = np.array(prev_poses2d_pred_ordered).copy()
    for ti, pi in zip(true_indices, pred_indices):
        result[ti] = poses3d_pred[pi]
        poses2d_pred_ordered[ti] = poses2d_pred[pi]

    return result, poses2d_pred_ordered


def nms_pose(poses, confs):
    order = np.argsort(-confs)
    poses = np.array(poses)[order]
    confs = np.array(confs)[order]
    resulting_poses = []
    resulting_confs = []
    for pose, conf in zip(poses, confs):
        for resulting_pose in resulting_poses:
            score = auc_for_nms(pose, resulting_pose, 300, 5)
            if score > 0.4:
                break
        else:
            resulting_poses.append(pose)
            resulting_confs.append(conf)
    return resulting_poses, resulting_confs


def smooth(tracks):
    if FLAGS.causal_smoothing:
        kernel = np.array([6, 2, 1]) / 9
        return scipy.ndimage.convolve1d(tracks, kernel, axis=1, origin=-1)

    kernel = np.array([1, 2, 6, 2, 1]) / 12
    return scipy.ndimage.convolve1d(tracks, kernel, axis=1)


def complete_track(track, n_frames):
    track_dict = dict(track)
    result = []
    for i in range(n_frames):
        if i in track_dict:
            result.append(track_dict[i])
        elif result:
            result.append(result[-1])
        else:
            result.append(np.full_like(track[0][1], fill_value=np.nan))
    return result


def save_result_file(seq_name, pred_dir, tracks):
    seq_filepaths = glob.glob(f'{paths.DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl')
    seq_path = next(p for p in seq_filepaths if os.path.basename(p) == f'{seq_name}.pkl')
    rel_path = '/'.join(util.split_path(seq_path)[-2:])
    out_path = f'{pred_dir}/{rel_path}'
    n_frames = len(glob.glob(f'{paths.DATA_ROOT}/3dpw/imageFiles/{seq_name}/image_*.jpg'))
    coords3d_raw = np.array([complete_track(track, n_frames) for track in tracks]) / 1000
    util.dump_pickle(dict(jointPositions=coords3d_raw), out_path)


def frames_of(video_path):
    with imageio.get_reader(video_path, 'ffmpeg') as reader:
        yield from reader


def main_visualize(q, joint_info):
    from mayavi import mlab
    import poseviz.image_viz
    import poseviz.init
    import poseviz.main_viz
    import poseviz.mayavi_util
    poseviz.init.initialize_simple()
    poseviz.image_viz.draw_checkerboard(floor_height=-1000)
    poseviz.mayavi_util.set_world_up([0, -1, 0])
    mv = poseviz.main_viz.MainViz(joint_info, joint_info, joint_info, 1, 'bird', True)

    @mlab.animate(delay=10, ui=False)
    def anim():
        initialized = False
        while True:
            try:
                image, poses, camera = q.get_nowait()
            except queue.Empty:
                yield
                continue
            mv.update(camera, image, poses)
            if not initialized:
                pivot = np.mean(poses, axis=(0, 1))
                camera_view = camera.copy()
                camera_view.t = (camera_view.t - pivot) * 1.5 + pivot
                camera_view.orbit_around(pivot, np.deg2rad(20), 'vertical')
                camera_view.orbit_around(pivot, np.deg2rad(-10), 'horizontal')
                poseviz.mayavi_util.set_view_to_camera(
                    camera_view, pivot=pivot, image_size=(image.shape[1], image.shape[0]),
                    allow_roll=False)
                initialized = True
            yield

    _ = anim()
    mlab.show()


if __name__ == '__main__':
    # Precomputed on non-3DPW data (on the combined merged dataset
    # that the large MeTRAbs model was trained on)
    mean_bone_lengths = np.array(
        [83.94, 125.16, 98.03, 264.69, 247.45, 87.99, 125.50, 103.76, 258.22, 250.64, 87.43, 220.48,
         66.97, 137.54, 114.53, 115.78, 381.26, 400.98, 150.62, 113.86, 389.31, 401.04, 151.03])

    ji = data.datasets3d.JointInfo(
        joints='pelv,lhip,rhip,bell,lkne,rkne,spin,lank,rank,thor,ltoe,rtoe,neck,lcla,rcla,head,'
               'lsho,rsho,lelb,relb,lwri,rwri,lhan,rhan',
        edges='head-neck-lcla-lsho-lelb-lwri-lhan,neck-rcla-rsho-relb-rwri-rhan,'
              'neck-thor-spin-bell-pelv-lhip-lkne-lank-ltoe,pelv-rhip-rkne-rank-rtoe')
    ji2d = data.datasets3d.JointInfo(
        'nose,neck,rsho,relb,rwri,lsho,lelb,lwri,rhip,rkne,rank,lhip,lkne,lank,reye,leye,lear,rear')
    main()
