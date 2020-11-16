import functools

import logging
import numpy as np
import os.path

import paths
import util
from data.joint_info import JointInfo
from util import TEST, TRAIN, VALID


class Pose3DDataset:
    def __init__(
            self, joint_info, train_examples=None, valid_examples=None, test_examples=None):
        self.joint_info = joint_info
        self.examples = {
            TRAIN: train_examples or [],
            VALID: valid_examples or [],
            TEST: test_examples or []}

        self.update_bones()

    def get_mean_bones(self, examples):
        coords3d = np.stack([ex.world_coords for ex in examples], axis=0)
        return [
            np.nanmean(np.linalg.norm(coords3d[:, i] - coords3d[:, j], axis=-1))
            for i, j in self.joint_info.stick_figure_edges]

    def update_bones(self):
        trainval_examples = [*self.examples[TRAIN], *self.examples[VALID]]
        if trainval_examples:
            self.trainval_bones = self.get_mean_bones(trainval_examples)
        if self.examples[TRAIN]:
            self.train_bones = self.get_mean_bones(self.examples[TRAIN])


class Pose3DExample:
    def __init__(
            self, image_path, world_coords, bbox, camera, *,
            activity_name='unknown', scene_name='unknown', mask=None, univ_coords=None):
        self.image_path = image_path
        self.world_coords = world_coords
        self.univ_coords = univ_coords if univ_coords is not None else None
        self.bbox = np.asarray(bbox)
        self.camera = camera
        self.activity_name = activity_name
        self.scene_name = scene_name
        self.mask = mask


def make_h36m_incorrect_S9(*args, **kwargs):
    import data.h36m
    return data.h36m.make_h36m(*args, **kwargs, correct_S9=False)


def make_h36m(*args, **kwargs):
    import data.h36m
    return data.h36m.make_h36m(*args, **kwargs)


def make_h36m_partial(*args, **kwargs):
    import data.h36m
    return data.h36m.make_h36m(*args, **kwargs, partial_visibility=True)


def make_mpi_inf_3dhp():
    import data.mpi_inf_3dhp
    return data.mpi_inf_3dhp.make_mpi_inf_3dhp()


def make_mpi_inf_3dhp_correctedTS6():
    import data.mpi_inf_3dhp
    return data.mpi_inf_3dhp.make_mpi_inf_3dhp(ts6_corr=True)


def make_muco():
    import data.muco
    return data.muco.make_muco()


def make_mupots():
    import data.mupots
    return data.mupots.make_mupots()


@util.cache_result_on_disk(
    f'{paths.CACHE_DIR}/muco_17_150k_old.pkl', min_time="2020-06-29T21:16:09")
def make_muco_17_150k_old():
    ds = util.load_pickle(f'{paths.CACHE_DIR}/muco_150k_old.pkl')
    mupots = make_mupots()
    convert_dataset(ds, mupots.joint_info)
    ds.examples[TEST] = mupots.examples[TEST]
    ds.update_bones()
    return ds


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/muco_17_150k.pkl', min_time="2020-06-29T21:16:09")
def make_muco_17_150k():
    ds = make_muco()

    # Take only the first 150,000 composite images
    def get_image_id(ex):
        return int(os.path.basename(ex.image_path).split('_')[0])

    ds.examples[TRAIN] = [e for e in ds.examples[0] if get_image_id(e) < 150000]

    # Take only the 17 MuPoTS joints
    mupots = make_mupots()
    convert_dataset(ds, mupots.joint_info)
    ds.examples[TEST] = mupots.examples[TEST]
    ds.update_bones()
    return ds


def make_many():
    joint_names = ['lhip', 'rhip', 'bell', 'lkne', 'rkne', 'spin', 'lank', 'rank', 'thor', 'ltoe',
                   'rtoe', 'neck', 'lcla', 'rcla', 'head', 'lsho', 'rsho', 'lelb', 'relb', 'lwri',
                   'rwri', 'lhan', 'rhan', 'pelv', 'head_h36m', 'head_muco', 'head_sailvos',
                   'htop_h36m', 'htop_muco', 'htop_sailvos', 'lcla_muco', 'lear', 'leye',
                   'lfin_h36m', 'lfoo_h36m', 'lfoo_muco', 'lhan_muco', 'lhip_cmu_panoptic',
                   'lhip_h36m', 'lhip_muco', 'lhip_sailvos', 'lsho_cmu_panoptic', 'lsho_h36m',
                   'lsho_muco', 'lsho_sailvos', 'lthu_h36m', 'neck_cmu_panoptic', 'neck_h36m',
                   'neck_muco', 'neck_sailvos', 'nose', 'pelv_cmu_panoptic', 'pelv_h36m',
                   'pelv_muco', 'pelv_sailvos', 'rcla_muco', 'rear', 'reye', 'rfin_h36m',
                   'rfoo_h36m', 'rfoo_muco', 'rhan_muco', 'rhip_cmu_panoptic', 'rhip_h36m',
                   'rhip_muco', 'rhip_sailvos', 'rsho_cmu_panoptic', 'rsho_h36m', 'rsho_muco',
                   'rsho_sailvos', 'rthu_h36m', 'spi2_muco', 'spi4_muco']
    edges = [(0, 3), (0, 23), (1, 4), (1, 23), (2, 5), (2, 23), (3, 6), (3, 37), (3, 38), (3, 39),
             (3, 40), (4, 7), (4, 62), (4, 63), (4, 64), (4, 65), (5, 8), (5, 47), (5, 49), (5, 52),
             (5, 53), (5, 54), (5, 71), (6, 9), (6, 34), (6, 35), (7, 10), (7, 59), (7, 60),
             (8, 11), (8, 71), (8, 72), (9, 34), (9, 35), (10, 59), (10, 60), (11, 12), (11, 13),
             (11, 14), (12, 15), (13, 16), (15, 17), (16, 18), (17, 19), (17, 41), (17, 42),
             (17, 43), (17, 44), (18, 20), (18, 66), (18, 67), (18, 68), (18, 69), (19, 21),
             (19, 33), (19, 36), (19, 45), (20, 22), (20, 58), (20, 61), (20, 70), (24, 27),
             (24, 47), (25, 28), (25, 48), (26, 29), (26, 49), (30, 43), (30, 48), (31, 32),
             (32, 50), (37, 51), (38, 52), (39, 53), (40, 54), (41, 46), (42, 47), (44, 49),
             (46, 50), (46, 51), (46, 66), (47, 67), (48, 55), (48, 72), (49, 69), (50, 57),
             (51, 62), (52, 63), (53, 64), (54, 65), (55, 68), (56, 57)]
    joint_info = JointInfo(joint_names, edges)
    import imageio
    import tempfile
    import cameralib
    _, image_path = tempfile.mkstemp(suffix='.jpg')
    imageio.imwrite(image_path, np.zeros((256, 256), dtype=np.uint8))
    dummy_example = Pose3DExample(
        image_path, np.zeros((joint_info.n_joints, 3), np.float32),
        [0, 0, 256, 256], cameralib.Camera())
    return Pose3DDataset(joint_info, [dummy_example], [dummy_example], [dummy_example])


def convert_dataset(src_dataset, dst_joint_info):
    mapping = get_coord_mapping(src_dataset.joint_info, dst_joint_info)
    src_dataset.examples = {
        phase: convert_examples(src_dataset.examples[phase], mapping)
        for phase in (TRAIN, VALID, TEST)}
    src_dataset.joint_info = dst_joint_info


def convert_examples(src_examples, mapping):
    return [convert_example(e, mapping) for e in util.progressbar(src_examples)]


def convert_coords(coords, mapping):
    coords_transf = np.einsum('jc,ij->ic', np.nan_to_num(coords), mapping)
    isnan_transf = np.einsum('jc,ij->ic', np.isnan(coords).astype(np.float32), mapping).astype(bool)
    coords_transf[isnan_transf] = np.nan
    return coords_transf


def convert_example(src_ex, mapping):
    src_ex.world_coords = convert_coords(src_ex.world_coords, mapping)
    if src_ex.univ_coords is not None:
        src_ex.univ_coords = convert_coords(src_ex.univ_coords, mapping)
    return src_ex


def get_coord_mapping(src_joint_info, dst_joint_info):
    """Returns a new coordinate array that can be indexed according to `dst_joint_info`.
    If a joint is in src but not in dst, it's thrown away, if a joint is in dst but not in
    src, then the corresponding values are set to NaN.
    """
    src_names = src_joint_info.names
    dst_names = dst_joint_info.names

    compatible_alternatives = {'tors': ['tors', 'spin'], 'spin': ['tors', 'spin']}

    mapping = np.zeros([dst_joint_info.n_joints, src_joint_info.n_joints])
    for i_dst, name in enumerate(dst_names):
        sought_names = compatible_alternatives.get(name, [name])
        found_names = list(set(sought_names) & set(src_names))
        if found_names:
            i_src = src_names.index(found_names[0])
            mapping[i_dst, i_src] = 1
        else:
            mapping[i_dst, 0] = np.nan
    return mapping


@functools.lru_cache()
def get_dataset(dataset_name):
    from options import FLAGS

    if dataset_name.endswith('.pkl'):
        return util.load_pickle(util.ensure_absolute_path(dataset_name))
    logging.debug(f'Making dataset {dataset_name}...')

    kwargs = {}

    def string_to_intlist(string):
        return tuple(int(s) for s in string.split(','))

    for subj_key in ['train_subjects', 'valid_subjects', 'test_subjects']:
        if hasattr(FLAGS, subj_key) and getattr(FLAGS, subj_key):
            kwargs[subj_key] = string_to_intlist(getattr(FLAGS, subj_key))

    return globals()[f'make_{dataset_name}'](**kwargs)
