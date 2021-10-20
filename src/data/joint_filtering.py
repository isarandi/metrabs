import numpy as np

import util
from tfu import TEST, TRAIN, VALID


def convert_dataset(src_dataset, dst_joint_info):
    mapping = get_coord_mapping(src_dataset.joint_info, dst_joint_info)
    src_dataset.examples = {
        phase: convert_examples(src_dataset.examples[phase], mapping)
        for phase in (TRAIN, VALID, TEST)}

    src_dataset.joint_info = dst_joint_info
    try:
        src_dataset.update_bones()
    except AttributeError:
        pass
    return src_dataset


def convert_examples(src_examples, mapping):
    return [convert_example(e, mapping) for e in util.progressbar(src_examples)]


def convert_coords(coords, mapping):
    try:
        coords_transf = np.einsum('jc,ij->ic', np.nan_to_num(coords), mapping)
    except:
        print(coords.shape, mapping.shape)
        raise
    isnan_transf = np.einsum('jc,ij->ic', np.isnan(coords).astype(np.float32), mapping).astype(bool)
    coords_transf[isnan_transf] = np.nan
    return coords_transf


def convert_example(src_ex, mapping):
    if hasattr(src_ex, 'coords'):
        src_ex.coords = convert_coords(src_ex.coords, mapping)
    if hasattr(src_ex, 'world_coords'):
        src_ex.world_coords = convert_coords(src_ex.world_coords, mapping)
    if hasattr(src_ex, 'univ_coords') and src_ex.univ_coords is not None:
        src_ex.univ_coords = convert_coords(src_ex.univ_coords, mapping)
    return src_ex


def get_coord_mapping(src_joint_info, dst_joint_info, suffix=''):
    """Returns a new coordinate array that can be indexed according to `dst_joint_info`.
    If a joint is in src but not in dst, it's thrown away, if a joint is in dst but not in
    src, then the corresponding values are set to NaN.
    """
    src_names = src_joint_info.names
    dst_names = dst_joint_info.names
    compatible_alternatives = {
        'tors': ['tors' + suffix, 'spin' + suffix, 'tors', 'spin'],
        'spin': ['tors' + suffix, 'spin' + suffix, 'tors', 'spin']}

    mapping = np.zeros([dst_joint_info.n_joints, src_joint_info.n_joints])
    for i_dst, name in enumerate(dst_names):
        sought_names = compatible_alternatives.get(name, [name + suffix, name])
        found_names = [n for n in sought_names if n in src_names]

        if found_names:
            i_src = src_names.index(found_names[0])
            mapping[i_dst, i_src] = 1
        else:
            mapping[i_dst, 0] = np.nan
    return mapping
