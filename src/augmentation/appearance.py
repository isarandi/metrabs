import numpy as np

import augmentation.background
import augmentation.color
import augmentation.voc_loader
import improc
import util
from options import FLAGS
from tfu import TRAIN


def augment_appearance(im, learning_phase, rng):
    occlusion_rng = util.new_rng(rng)
    color_rng = util.new_rng(rng)

    if learning_phase == TRAIN or FLAGS.test_aug:
        if FLAGS.occlude_aug_prob > 0:
            occlude_type = str(occlusion_rng.choice(['objects', 'random-erase']))
        else:
            occlude_type = None

        if occlude_type == 'objects':
            # For object occlusion augmentation, do the occlusion first, then the filtering,
            # so that the occluder blends into the image better.
            if occlusion_rng.uniform(0.0, 1.0) < FLAGS.occlude_aug_prob:
                im = object_occlude(im, occlusion_rng, inplace=True)
            if FLAGS.color_aug:
                im = augmentation.color.augment_color(im, color_rng)
        elif occlude_type == 'random-erase':
            # For random erasing, do color aug first, to keep the random block distributed
            # uniformly in 0-255, as in the Random Erasing paper
            if FLAGS.color_aug:
                im = augmentation.color.augment_color(im, color_rng)
            if occlude_type and occlusion_rng.uniform(0.0, 1.0) < FLAGS.occlude_aug_prob:
                im = random_erase(im, 0, 1 / 3, 0.3, 1.0 / 0.3, occlusion_rng, inplace=True)

    return im


def object_occlude(im, rng, inplace=True):
    # Following [Sárándi et al., arxiv:1808.09316, arxiv:1809.04987]
    factor = im.shape[0] / 256
    count = rng.randint(1, 8)
    occluders = augmentation.voc_loader.load_occluders()

    for i in range(count):
        occluder, occ_mask = util.choice(occluders, rng)
        rescale_factor = rng.uniform(0.2, 1.0) * factor * FLAGS.occlude_aug_scale

        occ_mask = improc.resize_by_factor(occ_mask, rescale_factor)
        occluder = improc.resize_by_factor(occluder, rescale_factor)

        center = rng.uniform(0, im.shape[0], size=2)
        im = improc.paste_over(occluder, im, alpha=occ_mask, center=center, inplace=inplace)

    return im


def random_erase(im, area_factor_low, area_factor_high, aspect_low, aspect_high, rng, inplace=True):
    # Following the random erasing paper [Zhong et al., arxiv:1708.04896]
    image_area = FLAGS.proc_side ** 2
    while True:
        occluder_area = (
                rng.uniform(area_factor_low, area_factor_high) *
                image_area * FLAGS.occlude_aug_scale)
        aspect_ratio = rng.uniform(aspect_low, aspect_high)
        height = (occluder_area * aspect_ratio) ** 0.5
        width = (occluder_area / aspect_ratio) ** 0.5
        pt1 = rng.uniform(0, FLAGS.proc_side, size=2)
        pt2 = pt1 + np.array([width, height])
        if np.all(pt2 < FLAGS.proc_side):
            pt1 = pt1.astype(int)
            pt2 = pt2.astype(int)
            if not inplace:
                im = im.copy()
            im[pt1[1]:pt2[1], pt1[0]:pt2[0]] = rng.randint(
                0, 255, size=(pt2[1] - pt1[1], pt2[0] - pt1[0], 3), dtype=im.dtype)
            return im
