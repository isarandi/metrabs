import itertools
import os.path
import re

import h5py
import numpy as np

import boxlib
import cameralib
import data.datasets3d as p3ds
import matlabfile
import paths
import util
from data.preproc_for_efficiency import make_efficient_example


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/3dhp.pkl', min_time="2020-11-02T22:14:33")
def make_mpi_inf_3dhp(camera_ids=(0, 1, 2, 4, 5, 6, 7, 8)):
    all_short_names = (
        'spi3,spi4,spi2,spin,pelv,neck,head,htop,lcla,lsho,lelb,lwri,lhan,rcla,rsho,relb,rwri,'
        'rhan,lhip,lkne,lank,lfoo,ltoe,rhip,rkne,rank,rfoo,rtoe'.split(','))

    test_set_selected_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 14]
    selected_joints = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 3, 6, 4]
    joint_names = [all_short_names[j] for j in selected_joints]

    edges = ('htop-head-neck-lsho-lelb-lwri,neck-rsho-relb-rwri,neck-spin-pelv-lhip-lkne-lank,'
             'pelv-rhip-rkne-rank')
    joint_info = p3ds.JointInfo(joint_names, edges)

    root_3dhp = f'{paths.DATA_ROOT}/3dhp'
    detections_all = util.load_pickle(f'{paths.DATA_ROOT}/3dhp/yolov3_person_detections.pkl')

    #################################
    # TRAINING AND VALIDATION SET
    #################################
    num_frames = np.asarray(
        [[6416, 12430], [6502, 6081], [12488, 12283], [6171, 6675], [12820, 12312], [6188, 6145],
         [6239, 6320], [6468, 6054]])

    train_subjects = [0, 1, 2, 3, 4, 5, 6]
    valid_subjects = [7]  # this is my own arbitrary split for validation (Istvan Sarandi)
    train_examples = []
    valid_examples = []

    pool = util.BoundedPool(None, 120)
    for i_subject, i_seq, i_cam in itertools.product(
            train_subjects + valid_subjects, range(2), camera_ids):
        seqpath = f'{root_3dhp}/S{i_subject + 1}/Seq{i_seq + 1}'
        print(f'Processing {seqpath} camera {i_cam}')

        cam3d_coords = [ann.reshape([ann.shape[0], -1, 3])[:, selected_joints]
                        for ann in matlabfile.load(f'{seqpath}/annot.mat')['annot3']]
        univ_cam3d_coords = [ann.reshape([ann.shape[0], -1, 3])[:, selected_joints]
                             for ann in matlabfile.load(f'{seqpath}/annot.mat')['univ_annot3']]
        cameras = load_cameras(f'{seqpath}/camera.calibration')

        examples_container = train_examples if i_subject in train_subjects else valid_examples
        frame_step = 5

        prev_coords = None
        camera = cameras[i_cam]
        n_frames = num_frames[i_subject, i_seq]

        if i_subject == 5 and i_seq == 1 and i_cam == 2:
            # This video is shorter for some reason
            n_frames = 3911

        for i_frame in util.progressbar(range(0, n_frames, frame_step)):
            image_relpath = (
                    f'3dhp/S{i_subject + 1}/Seq{i_seq + 1}/'
                    f'imageSequence/img_{i_cam}_{i_frame:06d}.jpg')

            cam_coords = cam3d_coords[i_cam][i_frame]
            world_coords = cameras[i_cam].camera_to_world(cam_coords)

            univ_camcoords = univ_cam3d_coords[i_cam][i_frame]
            univ_world_coords = cameras[i_cam].camera_to_world(univ_camcoords)

            # Check if the joints are within the image frame bounds
            if not np.all(camera.is_visible(world_coords, [2048, 2048])):
                continue

            im_coords = camera.camera_to_image(cam_coords)
            bbox = get_bbox(im_coords, image_relpath, detections_all)

            # Adaptive temporal sampling
            if (prev_coords is not None and
                    np.all(np.linalg.norm(world_coords - prev_coords, axis=1) < 100)):
                continue
            prev_coords = world_coords

            mask_path = image_relpath.replace('imageSequence', 'FGmasks')
            new_image_relpath = image_relpath.replace('3dhp', '3dhp_downscaled')
            ex = p3ds.Pose3DExample(
                image_relpath, world_coords, bbox, camera, mask=mask_path,
                univ_coords=univ_world_coords)

            pool.apply_async(make_efficient_example, (ex, new_image_relpath, 1, True),
                             callback=examples_container.append)

    print('Waiting for tasks...')
    pool.close()
    pool.join()
    print('Done...')
    #################################
    # TEST SET
    #################################
    test_examples = []

    cam1_4 = make_3dhp_test_camera(
        sensor_size=np.array([10, 10]), im_size=np.array([2048, 2048]), focal_length=7.32506,
        pixel_aspect=1.00044, center_offset=np.array([-0.0322884, 0.0929296]), distortion=None,
        origin=np.array([3427.28, 1387.86, 309.42]), up=np.array([-0.208215, 0.976233, 0.06014]),
        right=np.array([0.000575281, 0.0616098, -0.9981]))

    cam5_6 = make_3dhp_test_camera(
        sensor_size=np.array([10, 5.625]), im_size=np.array([1920, 1080]), focal_length=8.770747185,
        pixel_aspect=0.993236423, center_offset=np.array([-0.104908645, 0.104899704]),
        distortion=np.array([-0.276859611, 0.131125256, -0.000360494, -0.001149441, -0.049318332]),
        origin=np.array([-2104.3074, 1038.6707, -4596.6367]),
        up=np.array([0.025272345, 0.995038509, 0.096227370]),
        right=np.array([-0.939647257, -0.009210289, 0.342020929]))

    activity_names = [
        'Stand/Walk', 'Exercise', 'Sit on Chair', 'Reach/Crouch', 'On Floor', 'Sports', 'Misc.']
    for i_subject in range(1, 7):
        seqpath = f'{root_3dhp}/TS{i_subject}'
        annotation_path = f'{seqpath}/annot_data.mat'

        with h5py.File(annotation_path, 'r') as m:
            cam3d_coords = np.array(m['annot3'])[:, 0, test_set_selected_joints]
            univ_cam3d_coords = np.array(m['univ_annot3'])[:, 0, test_set_selected_joints]
            valid_frames = np.where(m['valid_frame'][:, 0])[0]
            activity_ids = m['activity_annotation'][:, 0].astype(int) - 1

        camera = cam1_4 if i_subject <= 4 else cam5_6
        scene = ['green-screen', 'no-green-screen', 'outdoor'][(i_subject - 1) // 2]

        for i_frame in valid_frames:
            image_relpath = f'3dhp/TS{i_subject}/imageSequence/img_{i_frame + 1:06d}.jpg'
            cam_coords = cam3d_coords[i_frame]
            univ_camcoords = univ_cam3d_coords[i_frame]
            activity = activity_names[activity_ids[i_frame]]
            world_coords = camera.camera_to_world(cam_coords)
            univ_world_coords = camera.camera_to_world(univ_camcoords)
            im_coords = camera.camera_to_image(cam_coords)
            bbox = get_bbox(im_coords, image_relpath, detections_all)

            ex = p3ds.Pose3DExample(
                image_relpath, world_coords, bbox, camera, activity_name=activity,
                scene_name=scene, univ_coords=univ_world_coords)
            test_examples.append(ex)

    train_examples.sort(key=lambda x: x.image_path)
    valid_examples.sort(key=lambda x: x.image_path)
    test_examples.sort(key=lambda x: x.image_path)
    return p3ds.Pose3DDataset(joint_info, train_examples, valid_examples, test_examples)


def get_bbox(im_coords, image_relpath, detections_all):
    joint_box = boxlib.expand(boxlib.bb_of_points(im_coords), 1.05)
    relpath_in_dataset = os.path.join(*util.split_path(image_relpath)[1:])
    if relpath_in_dataset in detections_all and detections_all[relpath_in_dataset]:
        most_confident_detection = max(detections_all[relpath_in_dataset], key=lambda x: x[4])
        detection_box = np.array(most_confident_detection[:4])
        union_box = boxlib.box_hull(detection_box, joint_box)
        # Sanity check
        if boxlib.iou(union_box, joint_box) > 0.5:
            return union_box
    return joint_box


def load_cameras(camcalib_path):
    def to_array(string):
        return np.array([float(p) for p in string.split(' ')])

    def make_camera_from_match(match):
        intrinsic_matrix = np.reshape(to_array(match['intrinsic']), [4, 4])[:3, :3]
        extrinsic_matrix = np.reshape(to_array(match['extrinsic']), [4, 4])
        R = extrinsic_matrix[:3, :3]
        optical_center = -R.T @ extrinsic_matrix[:3, 3]
        return cameralib.Camera(optical_center, R, intrinsic_matrix, None, world_up=(0, 1, 0))

    camcalib_text = util.read_file(camcalib_path)
    pattern = (
            r'name\s+\d+\n'
            r'  sensor\s+\d+ \d+\n'
            r'  size\s+\d+ \d+\n' +
            r'  animated\s+\d\n\s+intrinsic\s+(?P<intrinsic>(-?[0-9.]+\s*?){16})\s*\n'
            r'  extrinsic\s+(?P<extrinsic>(-?[0-9.]+\s*?){16})\s*\n'
            r'  radial\s+\d')
    return [make_camera_from_match(m) for m in re.finditer(pattern, camcalib_text)]


def make_3dhp_test_camera(
        sensor_size, im_size, focal_length, pixel_aspect, center_offset, distortion, origin, up,
        right):
    R = np.row_stack([right, -up, np.cross(up, right)])
    intrinsic_matrix = np.diag([focal_length, focal_length, 1])
    intrinsic_matrix[:2, 2] = sensor_size / 2 + center_offset
    mm_to_px_factors = im_size / sensor_size * np.array([1, pixel_aspect])
    intrinsic_matrix[:2] = np.diag(mm_to_px_factors) @ intrinsic_matrix[:2]
    return cameralib.Camera(origin, R, intrinsic_matrix, distortion, world_up=(0, 1, 0))
