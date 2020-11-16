import functools
import os.path
import re

import imageio
import numpy as np
import scipy.optimize

import boxlib
import cameralib
import data.datasets3d as p3ds
import improc
import matlabfile
import paths
import util


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/muco.pkl', min_time="2020-02-24T19:18:24")
def make_muco():
    joint_info, selected_joints = make_joint_info()

    root_3dhp = f'{paths.DATA_ROOT}/3dhp'
    root_muco = f'{paths.DATA_ROOT}/muco'
    sample_info = np.load(f'{root_muco}/composite_frame_origins.npy')
    n_all_joints = 28
    valid_indices = list(np.load(f'{root_muco}/valid_composite_frame_indices.npy'))
    all_detections = util.load_pickle(f'{root_muco}/yolov3_detections.pkl')
    all_detections = np.array([all_detections[k] for k in sorted(all_detections.keys())])
    all_visible_boxes = np.load(f'{root_muco}/visible_boxes.npy')
    matloader = functools.lru_cache(1024)(matlabfile.load)

    @functools.lru_cache(1024)
    def get_world_coords(i_subject, i_seq, i_cam, anno_name):
        seqpath = f'{root_3dhp}/S{i_subject}/Seq{i_seq}'
        anno_file = matloader(f'{seqpath}/annot.mat')
        camcoords = anno_file[anno_name][i_cam].reshape(
            [-1, n_all_joints, 3])[:, selected_joints]
        camera = load_cameras(f'{seqpath}/camera.calibration')[i_cam]
        world_coords = [camera.camera_to_world(c) for c in camcoords]
        return world_coords

    examples = []

    with util.BoundedPool(None, 120) as pool:
        for i_sample, people, detections, visible_boxes in zip(
                util.progressbar(valid_indices), sample_info[valid_indices],
                all_detections[valid_indices], all_visible_boxes[valid_indices]):

            detections = [box for box in detections if box[-1] > 0.1]
            if not detections:
                continue

            filename = f'{i_sample + 1:06d}.jpg'
            image_relpath = f'unaugmented_set_001/{filename[:2]}/{filename[:4]}/{filename}'

            gt_people = []
            for i_person, ((i_subject, i_seq, i_cam, i_frame), visible_box) in enumerate(
                    zip(people, visible_boxes)):
                seqpath = f'{root_3dhp}/S{i_subject}/Seq{i_seq}'
                world_coords = get_world_coords(i_subject, i_seq, i_cam, 'annot3')[i_frame]
                univ_world_coords = get_world_coords(
                    i_subject, i_seq, i_cam, 'univ_annot3')[i_frame]
                camera = load_cameras(f'{seqpath}/camera.calibration')[i_cam]

                im_coords = camera.world_to_image(world_coords)
                coord_bbox = boxlib.expand(boxlib.intersect(
                    boxlib.bb_of_points(im_coords),
                    boxlib.full_box([2048, 2048])), 1.05)
                bbox = boxlib.intersect_vertical(visible_box, coord_bbox)

                ex = p3ds.Pose3DExample(
                    image_relpath, world_coords, bbox, camera, mask=None,
                    univ_coords=univ_world_coords)
                gt_people.append(ex)

            if not gt_people:
                continue

            iou_matrix = np.array([[boxlib.iou(gt_person.bbox, box[:4])
                                    for box in detections]
                                   for gt_person in gt_people])
            gt_indices, det_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)

            for i_gt, i_det in zip(gt_indices, det_indices):
                gt_box = gt_people[i_gt].bbox
                det_box = detections[i_det]
                if (iou_matrix[i_gt, i_det] > 0.1 and
                        boxlib.area(det_box) < 2 * boxlib.area(gt_box)):
                    ex = gt_people[i_gt]
                    ex.bbox = np.array(detections[i_det][:4])
                    pool.apply_async(make_efficient_example, (ex, root_muco, i_gt),
                                     callback=examples.append)

    examples.sort(key=lambda ex: ex.image_path)
    return p3ds.Pose3DDataset(joint_info, examples)


def make_efficient_example(ex, root_muco, i_person):
    image_relpath = ex.image_path
    max_rotate = np.pi / 6
    padding_factor = 1 / 0.85
    scale_up_factor = 1 / 0.85
    scale_down_factor = 1 / 0.85
    shift_factor = 1.2
    base_dst_side = 256
    box_center = boxlib.center(ex.bbox)
    s = np.sin(max_rotate)
    c = np.cos(max_rotate)
    rot_bbox_size = (np.array([[c, s], [s, c]]) @ ex.bbox[2:, np.newaxis])[:, 0]
    side = np.max(rot_bbox_size)
    rot_bbox_size = np.array([side, side])
    rot_bbox = boxlib.box_around(box_center, rot_bbox_size)

    scale_factor = min(base_dst_side / np.max(ex.bbox[2:]) * scale_up_factor, 1)
    expansion_factor = padding_factor * shift_factor * scale_down_factor
    expanded_bbox = boxlib.expand(rot_bbox, expansion_factor)
    expanded_bbox = boxlib.intersect(expanded_bbox, boxlib.full_box([2048, 2048]))

    new_camera = ex.camera.copy()
    new_camera.intrinsic_matrix[:2, 2] -= expanded_bbox[:2]
    new_camera.scale_output(scale_factor)
    new_camera.undistort()

    dst_shape = improc.rounded_int_tuple(scale_factor * expanded_bbox[[3, 2]])
    new_im_path = f'{root_muco}_downscaled/{image_relpath[:-4]}_{i_person:01d}.jpg'
    if not (util.is_file_newer(new_im_path, "2020-02-15T23:28:26")):
        im = improc.imread_jpeg(f'{root_muco}/{image_relpath}')
        new_im = cameralib.reproject_image(im, ex.camera, new_camera, dst_shape, antialias_factor=4)
        util.ensure_path_exists(new_im_path)
        imageio.imwrite(new_im_path, new_im, quality=95)

    new_bbox_topleft = cameralib.reproject_image_points(ex.bbox[:2], ex.camera, new_camera)
    new_bbox = np.concatenate([new_bbox_topleft, ex.bbox[2:] * scale_factor])

    if ex.mask is None:
        noext, ext = os.path.splitext(image_relpath[:-4])
        noext = noext.replace('unaugmented_set_001/', '')
        mask = improc.decode_mask(util.load_pickle(f'{root_muco}/masks/{noext}.pkl'))
    else:
        mask = ex.mask

    if mask is False:
        new_mask_encoded = None
    else:
        new_mask = cameralib.reproject_image(mask, ex.camera, new_camera, dst_shape)
        new_mask_encoded = improc.encode_mask(new_mask)

    return p3ds.Pose3DExample(
        os.path.relpath(new_im_path, paths.DATA_ROOT), ex.world_coords.astype(np.float32),
        new_bbox.astype(np.float32), new_camera, mask=new_mask_encoded,
        univ_coords=ex.univ_coords.astype(np.float32))


@functools.lru_cache(1024)
def load_cameras(camcalib_path):
    def to_array(string):
        return np.array([float(p) for p in string.split(' ')])

    def make_camera_from_match(match):
        intrinsic_matrix = np.reshape(to_array(match['intrinsic']), [4, 4])[:3, :3]
        extrinsic_matrix = np.reshape(to_array(match['extrinsic']), [4, 4])
        R = extrinsic_matrix[:3, :3]
        eye = R.T @ extrinsic_matrix[:3, 3]
        return cameralib.Camera(
            eye, R, intrinsic_matrix, None, world_up=(0, 1, 0))

    camcalib_text = util.read_file(camcalib_path)
    pattern = (
            r'name\s+(?P<name>\d+)\n  sensor\s+(?P<sensor>\d+ \d+)\n  size\s+(?P<size>\d+ '
            r'\d+)\n ' +
            r' animated\s+(?P<animated>\d)\n\s+intrinsic\s+(?P<intrinsic>(-?[0-9.]+\s*?){' +
            r'16})\s*\n  extrinsic\s+(?P<extrinsic>(-?[0-9.]+\s*?){16})\s*\n  radial\s+(' +
            r'?P<radial>\d)')
    return [make_camera_from_match(m) for m in re.finditer(pattern, camcalib_text)]


def make_joint_info():
    all_joint_names = (
        'thor,spi4,spi2,spin,pelv,neck,head,htop,lcla,lsho,lelb,lwri,lhan,rcla,rsho,relb,rwri,'
        'rhan,lhip,lkne,lank,lfoo,ltoe,rhip,rkne,rank,rfoo,rtoe'.split(','))

    selected_joints = [
        0, 1, 2, 8, 12, 13, 17, 21, 22, 26, 27, 7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20,
        3, 6, 4]
    joint_names = [all_joint_names[j] for j in selected_joints]
    edges = (
        'htop-head-neck-lcla-lsho-lelb-lwri-lhan,neck-rcla-rsho-relb-rwri-rhan,'
        'neck-spi4-thor-spi2-spin-pelv-lhip-lkne-lank-lfoo-ltoe,pelv-rhip-rkne-rank-rfoo-rtoe')
    joint_info = p3ds.JointInfo(joint_names, edges)
    return joint_info, selected_joints
