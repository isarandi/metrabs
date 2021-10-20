import functools
import itertools
import os
import os.path
import xml.etree.ElementTree

import numpy as np
import spacepy.pycdf
import transforms3d

import cameralib
import data.datasets3d as ps3d
import paths
import util
from data.preproc_for_efficiency import make_efficient_example


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/h36m.pkl', min_time="2020-11-02T21:30:43")
def make_h36m(
        train_subjects=(1, 5, 6, 7, 8), valid_subjects=(), test_subjects=(9, 11),
        correct_S9=True, partial_visibility=False):
    joint_names = (
        'rhip,rkne,rank,lhip,lkne,lank,tors,neck,head,htop,'
        'lsho,lelb,lwri,rsho,relb,rwri,pelv'.split(','))
    edges = (
        'htop-head-neck-lsho-lelb-lwri,neck-rsho-relb-rwri,'
        'neck-tors-pelv-lhip-lkne-lank,pelv-rhip-rkne-rank')
    joint_info = ps3d.JointInfo(joint_names, edges)

    if not util.all_disjoint(train_subjects, valid_subjects, test_subjects):
        raise Exception('Set of train, val and test subject must be disjoint.')

    # use last subject of the non-test subjects for validation
    train_examples = []
    test_examples = []
    valid_examples = []
    pool = util.BoundedPool(None, 120)

    if partial_visibility:
        dir_suffix = '_partial'
        further_expansion_factor = 1.8
    else:
        dir_suffix = '' if correct_S9 else 'incorrect_S9'
        further_expansion_factor = 1

    for i_subject in [*test_subjects, *train_subjects, *valid_subjects]:
        if i_subject in train_subjects:
            examples_container = train_examples
        elif i_subject in valid_subjects:
            examples_container = valid_examples
        else:
            examples_container = test_examples

        frame_step = 5 if i_subject in train_subjects else 64

        for activity_name, camera_id in itertools.product(get_activity_names(i_subject), range(4)):
            print(f'Processing S{i_subject} {activity_name} {camera_id}')
            image_relpaths, world_coords_all, bboxes, camera = get_examples(
                i_subject, activity_name, camera_id, frame_step=frame_step, correct_S9=correct_S9)
            prev_coords = None
            for image_relpath, world_coords, bbox in zip(
                    util.progressbar(image_relpaths), world_coords_all, bboxes):
                # Using very similar examples is wasteful when training. Therefore:
                # skip frame if all keypoints are within a distance compared to last stored frame.
                # This is not done when testing, as it would change the results.
                if (i_subject in train_subjects and prev_coords is not None and
                        np.all(np.linalg.norm(world_coords - prev_coords, axis=1) < 100)):
                    continue
                prev_coords = world_coords
                activity_name = activity_name.split(' ')[0]
                ex = ps3d.Pose3DExample(
                    image_relpath, world_coords, bbox, camera, activity_name=activity_name)
                new_image_relpath = image_relpath.replace('h36m', f'h36m_downscaled{dir_suffix}')
                pool.apply_async(
                    make_efficient_example, (ex, new_image_relpath, further_expansion_factor),
                    callback=examples_container.append)

    print('Waiting for tasks...')
    pool.close()
    pool.join()
    print('Done...')
    train_examples.sort(key=lambda x: x.image_path)
    valid_examples.sort(key=lambda x: x.image_path)
    test_examples.sort(key=lambda x: x.image_path)
    return ps3d.Pose3DDataset(joint_info, train_examples, valid_examples, test_examples)


def correct_boxes(bboxes, path, world_coords, camera):
    """Three activties for subject S9 have erroneous bounding boxes, they are horizontally shifted.
    This function corrects them. Use --dataset=h36m-incorrect-S9 to use the erroneous annotation."""

    def correct_image_coords(bad_imcoords):
        root_depths = camera.world_to_camera(world_coords[:, -1])[:, 2:]
        bad_worldcoords = camera.image_to_world(bad_imcoords, camera_depth=root_depths)
        good_worldcoords = bad_worldcoords + np.array([-200, 0, 0])
        good_imcoords = camera.world_to_image(good_worldcoords)
        return good_imcoords

    if 'S9' in path and ('SittingDown 1' in path or 'Waiting 1' in path or 'Greeting.' in path):
        toplefts = correct_image_coords(bboxes[:, :2])
        bottomrights = correct_image_coords(bboxes[:, :2] + bboxes[:, 2:])
        return np.concatenate([toplefts, bottomrights - toplefts], axis=-1)

    return bboxes


def correct_world_coords(coords, path):
    """Three activties for subject S9 have erroneous coords, they are horizontally shifted.
    This corrects them. Use --dataset=h36m-incorrect-S9 to use the erroneous annotation."""
    if 'S9' in path and ('SittingDown 1' in path or 'Waiting 1' in path or 'Greeting.' in path):
        coords = coords.copy()
        coords[:, :, 0] -= 200
    return coords


def get_examples(i_subject, activity_name, i_camera, frame_step=5, correct_S9=True):
    camera_names = ['54138969', '55011271', '58860488', '60457274']
    camera_name = camera_names[i_camera]
    h36m_root = f'{paths.DATA_ROOT}/h36m/'
    camera = get_cameras(f'{h36m_root}/Release-v1.2/metadata.xml')[i_camera][i_subject - 1]

    def load_coords(path):
        with spacepy.pycdf.CDF(path) as cdf_file:
            coords_raw_all = np.array(cdf_file['Pose'], np.float32)[0]
        coords_raw = coords_raw_all[::frame_step]
        i_relevant_joints = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27, 0]
        coords_new_shape = [coords_raw.shape[0], -1, 3]
        return coords_raw_all.shape[0], coords_raw.reshape(coords_new_shape)[:, i_relevant_joints]

    pose_folder = f'{h36m_root}/S{i_subject}/MyPoseFeatures'
    coord_path = f'{pose_folder}/D3_Positions/{activity_name}.cdf'
    n_total_frames, world_coords = load_coords(coord_path)

    if correct_S9:
        world_coords = correct_world_coords(world_coords, coord_path)

    image_relfolder = f'h36m/S{i_subject}/Images/{activity_name}.{camera_name}'
    image_relpaths = [f'{image_relfolder}/frame_{i_frame:06d}.jpg'
                      for i_frame in range(0, n_total_frames, frame_step)]

    bbox_path = f'{h36m_root}/S{i_subject}/BBoxes/{activity_name}.{camera_name}.npy'
    bboxes = np.load(bbox_path)[::frame_step]
    if correct_S9:
        bboxes = correct_boxes(bboxes, bbox_path, world_coords, camera)

    return image_relpaths, world_coords, bboxes, camera


@functools.lru_cache()
def get_cameras(metadata_path):
    root = xml.etree.ElementTree.parse(metadata_path).getroot()
    cam_params_text = root.findall('w0')[0].text
    numbers = np.array([float(x) for x in cam_params_text[1:-1].split(' ')])
    extrinsic = numbers[:264].reshape(4, 11, 6)
    intrinsic = numbers[264:].reshape(4, 9)

    cameras = [[make_h36m_camera(extrinsic[i_camera, i_subject], intrinsic[i_camera])
                for i_subject in range(11)]
               for i_camera in range(4)]
    return cameras


def make_h36m_camera(extrinsic_params, intrinsic_params):
    x_angle, y_angle, z_angle = extrinsic_params[:3]
    R = transforms3d.euler.euler2mat(x_angle, y_angle, z_angle, 'rxyz')
    t = extrinsic_params[3:6]
    f, c, k, p = np.split(intrinsic_params, (2, 4, 7))
    distortion_coeffs = np.array([k[0], k[1], p[0], p[1], k[2]], np.float32)
    intrinsic_matrix = np.array([
        [f[0], 0, c[0]],
        [0, f[1], c[1]],
        [0, 0, 1]], np.float32)
    return cameralib.Camera(t, R, intrinsic_matrix, distortion_coeffs)


def get_activity_names(i_subject):
    h36m_root = f'{paths.DATA_ROOT}/h36m/'
    subject_images_root = f'{h36m_root}/S{i_subject}/Images/'
    subdirs = [elem for elem in os.listdir(subject_images_root)
               if os.path.isdir(f'{subject_images_root}/{elem}')]
    activity_names = set(elem.split('.')[0] for elem in subdirs if '_' not in elem)
    return sorted(activity_names)


def generate_poseviz_gt(i_subject, activity_name, camera_id):
    camera_names = ['54138969', '55011271', '58860488', '60457274']
    camera_name = camera_names[camera_id]
    image_relpaths, world_coords_all, bboxes, camera = get_examples(
        i_subject, activity_name, camera_id, frame_step=1, correct_S9=True)

    results = []
    examples = []
    for image_relpath, world_coords, bbox in zip(image_relpaths, world_coords_all, bboxes):
        results.append({
            'gt_poses': [world_coords.tolist()],
            'camera_intrinsics': camera.intrinsic_matrix.tolist(),
            'camera_extrinsics': camera.get_extrinsic_matrix().tolist(),
            'image_path': image_relpath,
            'bboxes': [bbox.tolist()]
        })
        ex = ps3d.Pose3DExample(
            image_relpath, world_coords, bbox, camera, activity_name=activity_name)
        examples.append(ex)

    joint_names = (
        'rhip,rkne,rank,lhip,lkne,lank,tors,neck,head,htop,'
        'lsho,lelb,lwri,rsho,relb,rwri,pelv'.split(','))
    edges = (
        'htop-head-neck-lsho-lelb-lwri,neck-rsho-relb-rwri,'
        'neck-tors-pelv-lhip-lkne-lank,pelv-rhip-rkne-rank')
    joint_info = ps3d.JointInfo(joint_names, edges)
    ds = ps3d.Pose3DDataset(joint_info, test_examples=examples)
    util.dump_pickle(
        ds, f'{paths.DATA_ROOT}/h36m/poseviz/S{i_subject}_{activity_name}_{camera_name}.pkl')

    output = {}
    output['joint_names'] = joint_info.names
    output['stick_figure_edges'] = joint_info.stick_figure_edges
    output['world_up'] = camera.world_up.tolist()
    output['frame_infos'] = results
    util.dump_json(
        output, f'{paths.DATA_ROOT}/h36m/poseviz/S{i_subject}_{activity_name}_{camera_name}.json')
