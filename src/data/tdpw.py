import glob
import os.path
import pickle

import boxlib
import cameralib
import data.datasets3d as p3ds
import numpy as np
import paths
import util
from data.preproc_for_efficiency import make_efficient_example


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/tdpw.pkl', min_time="2021-07-09T12:26:16")
def make_tdpw():
    root = '/globalwork/datasets/3DPW'
    body_joint_names = (
        'pelv,lhip,rhip,bell,lkne,rkne,spin,lank,rank,thor,ltoe,rtoe,neck,lcla,rcla,head,lsho,'
        'rsho,lelb,relb,lwri,rwri,lhan,rhan'.split(','))
    selected_joints = [*range(1, 24), 0]
    joint_names = [body_joint_names[j] for j in selected_joints]
    edges = ('head-neck-lcla-lsho-lelb-lwri-lhan,'
             'neck-rcla-rsho-relb-rwri-rhan,'
             'neck-thor-spin-bell-pelv-lhip-lkne-lank-ltoe,'
             'pelv-rhip-rkne-rank-rtoe')
    joint_info = p3ds.JointInfo(joint_names, edges)

    def get_examples(phase, pool):
        result = []
        seq_filepaths = glob.glob(f'{root}/sequenceFiles/{phase}/*.pkl')
        for filepath in seq_filepaths:
            with open(filepath, 'rb') as f:
                seq = pickle.load(f, encoding='latin1')
            seq_name = seq['sequence']
            intrinsics = seq['cam_intrinsics']
            extrinsics_per_frame = seq['cam_poses']

            for i_person, (coord_seq, coords2d_seq, trans_seq, camvalid_seq) in enumerate(zip(
                    seq['jointPositions'], seq['poses2d'], seq['trans'], seq['campose_valid'])):
                for i_frame, (coords, coords2d, trans, extrinsics, campose_valid) in enumerate(
                        zip(coord_seq, coords2d_seq, trans_seq, extrinsics_per_frame,
                            camvalid_seq)):
                    if not campose_valid or np.all(coords2d == 0):
                        continue

                    impath = f'{root}/imageFiles/{seq_name}/image_{i_frame:05d}.jpg'
                    camera = cameralib.Camera(
                        extrinsic_matrix=extrinsics, intrinsic_matrix=intrinsics,
                        world_up=(0, 1, 0))
                    camera.t *= 1000
                    world_coords = (coords.reshape(-1, 3))[selected_joints] * 1000
                    camera2 = cameralib.Camera(intrinsic_matrix=intrinsics, world_up=(0, -1, 0))
                    camcoords = camera.world_to_camera(world_coords)
                    imcoords = camera.world_to_image(world_coords)
                    bbox = boxlib.expand(boxlib.bb_of_points(imcoords), 1.15)
                    ex = p3ds.Pose3DExample(impath, camcoords, bbox=bbox, camera=camera2)
                    noext, ext = os.path.splitext(os.path.relpath(impath, root))
                    new_image_relpath = f'tdpw_downscaled/{noext}_{i_person:03d}.jpg'
                    pool.apply_async(
                        make_efficient_example,
                        (ex, new_image_relpath, 1, False, "2021-07-09T12:28:07"),
                        callback=result.append)
        return result

    with util.BoundedPool(None, 120) as pool:
        train_examples = get_examples('train', pool)
        val_examples = get_examples('validation', pool)
        test_examples = get_examples('test', pool)

    test_examples = [*train_examples, *val_examples, *test_examples]
    test_examples.sort(key=lambda ex: ex.image_path)
    return p3ds.Pose3DDataset(joint_info, None, None, test_examples)
