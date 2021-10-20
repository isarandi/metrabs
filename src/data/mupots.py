import os

import numpy as np
import scipy.optimize

import boxlib
import cameralib
import data.datasets3d as p3ds
import matlabfile
import paths
import util


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/mupots-yolo.pkl', min_time="2021-09-16T20:39:52")
def make_mupots():
    joint_names = (
        'htop,neck,rsho,relb,rwri,lsho,lelb,lwri,rhip,rkne,rank,lhip,lkne,lank,spin,head,pelv')
    edges = (
        'htop-head-neck-spin-pelv-lhip-lkne-lank,'
        'lwri-lelb-lsho-neck-rsho-relb-rwri,pelv-rhip-rkne-rank')
    joint_info = p3ds.JointInfo(joint_names, edges)

    root = f'{paths.DATA_ROOT}/mupots'
    intrinsic_matrices = util.load_json(f'{root}/camera_intrinsics.json')

    dummy_coords = np.ones((joint_info.n_joints, 3))
    detections_all = util.load_pickle(f'{root}/yolov3_detections.pkl')

    examples_test = []
    for i_seq in range(1, 21):
        annotations = matlabfile.load(f'{root}/TS{i_seq}/annot.mat')['annotations']
        intrinsic_matrix = intrinsic_matrices[f'TS{i_seq}']
        camera = cameralib.Camera(
            np.zeros(3), np.eye(3), intrinsic_matrix, distortion_coeffs=None, world_up=(0, -1, 0))

        n_frames = annotations.shape[0]
        for i_frame in range(n_frames):
            image_relpath = f'TS{i_seq}/img_{i_frame:06d}.jpg'
            detections_frame = detections_all[image_relpath]
            image_path = f'{root}/{image_relpath}'
            for detection in detections_frame:
                confidence = detection[4]
                if confidence > 0.1:
                    ex = p3ds.Pose3DExample(
                        os.path.relpath(image_path, paths.DATA_ROOT),
                        dummy_coords, detection[:4], camera,
                        mask=None, univ_coords=dummy_coords, scene_name=f'TS{i_seq}')
                    examples_test.append(ex)

    return p3ds.Pose3DDataset(joint_info, test_examples=examples_test)


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/mupots-yolo-val.pkl', min_time="2021-09-10T00:00:18")
def make_mupots_yolo():
    all_short_names = (
        'thor,spi4,spi2,spin,pelv,neck,head,htop,lcla,lsho,lelb,lwri,lhan,rcla,rsho,relb,rwri,'
        'rhan,lhip,lkne,lank,lfoo,ltoe,rhip,rkne,rank,rfoo,rtoe'.split(','))

    # originally: [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]
    selected_joints = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 3, 6, 4]
    order_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 14]
    joint_names = [all_short_names[j] for j in selected_joints]
    j = p3ds.JointInfo.make_id_map(joint_names)
    edges = [
        (j.htop, j.head), (j.head, j.neck), (j.neck, j.lsho), (j.lsho, j.lelb), (j.lelb, j.lwri),
        (j.neck, j.rsho), (j.rsho, j.relb), (j.relb, j.rwri), (j.neck, j.spin), (j.spin, j.pelv),
        (j.pelv, j.lhip), (j.lhip, j.lkne), (j.lkne, j.lank), (j.pelv, j.rhip), (j.rhip, j.rkne),
        (j.rkne, j.rank)]
    joint_info = p3ds.JointInfo(j, edges)

    root = f'{paths.DATA_ROOT}/mupots'
    intrinsic_matrices = util.load_json(f'{root}/camera_intrinsics.json')

    dummy_coords = np.ones((joint_info.n_joints, 3))
    detections_all = util.load_pickle(f'{root}/yolov3_detections.pkl')

    examples_val = []
    examples_test = []
    for i_seq in range(1, 21):
        annotations = matlabfile.load(f'{root}/TS{i_seq}/annot.mat')['annotations']
        intrinsic_matrix = intrinsic_matrices[f'TS{i_seq}']
        camera = cameralib.Camera(
            np.zeros(3), np.eye(3), intrinsic_matrix, distortion_coeffs=None, world_up=(0, -1, 0))

        n_people = annotations.shape[1]
        n_frames = annotations.shape[0]
        for i_frame in range(n_frames):

            image_relpath = f'TS{i_seq}/img_{i_frame:06d}.jpg'
            detections_frame = detections_all[image_relpath]
            image_path = f'{root}/{image_relpath}'
            for detection in detections_frame:
                if detection[4] > 0.1:
                    ex = p3ds.Pose3DExample(
                        image_path, dummy_coords, detection[:4], camera,
                        mask=None, univ_coords=dummy_coords, scene_name=f'TS{i_seq}')
                    examples_test.append(ex)

            gt_people = []

            for i_person in range(n_people):
                world_coords = np.array(
                    annotations[i_frame, i_person].annot3.T[order_joints], dtype=np.float32)
                univ_world_coords = np.array(
                    annotations[i_frame, i_person].univ_annot3.T[order_joints], dtype=np.float32)
                im_coords = camera.world_to_image(world_coords)
                gt_box = boxlib.expand(boxlib.bb_of_points(im_coords), 1.1)
                ex = p3ds.Pose3DExample(
                    image_path, world_coords, gt_box, camera,
                    mask=None, univ_coords=univ_world_coords, scene_name=f'TS{i_seq}')
                gt_people.append(ex)

            confident_detections = [det for det in detections_frame if det[-1] > 0.1]
            if confident_detections:
                iou_matrix = np.array([[boxlib.iou(gt_person.bbox, box[:4])
                                        for box in confident_detections]
                                       for gt_person in gt_people])
                gt_indices, detection_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)
                for i_gt, i_det in zip(gt_indices, detection_indices):
                    if iou_matrix[i_gt, i_det] > 0.1:
                        ex = gt_people[i_gt]
                        ex.bbox = np.array(confident_detections[i_det][:4])
                        examples_val.append(ex)

    return p3ds.Pose3DDataset(
        joint_info, valid_examples=examples_val, test_examples=examples_test)


def get_cameras(json_path):
    json_data = util.load_json(json_path)
    intrinsic_matrices = [json_data[f'TS{i_seq}'] for i_seq in range(1, 21)]
    return [cameralib.Camera(intrinsic_matrix=intrinsic_matrix, world_up=(0, -1, 0))
            for intrinsic_matrix in intrinsic_matrices]
