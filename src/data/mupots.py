import numpy as np
import os
import cameralib
import data.datasets3d as p3ds
import matlabfile
import paths
import util


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/mupots-yolo.pkl', min_time="2020-02-21T18:33:46")
def make_mupots():
    joint_names = (
        'htop,neck,rsho,relb,rwri,lsho,lelb,lwri,rhip,rkne,rank,lhip,lkne,lank,spin,head,pelv')
    edges = (
        'htop-head-neck-spin-pelv-lhip-lkne-lank,'
        'lwri-lelb-lsho-neck-rsho-relb-rwri,pelv-rhip-rkne-rank')
    joint_info = p3ds.JointInfo(joint_names, edges)

    #import data.muco
    #joint_info = data.muco.make_joint_info()[0]

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

    return p3ds.Pose3DDataset(joint_info, valid_examples=examples_val, test_examples=examples_test)


def get_cameras(json_path):
    json_data = util.load_json(json_path)
    intrinsic_matrices = [json_data[f'TS{i_seq}'] for i_seq in range(1, 21)]
    return [cameralib.Camera(intrinsic_matrix=intrinsic_matrix, world_up=(0, -1, 0))
            for intrinsic_matrix in intrinsic_matrices]
