#!/usr/bin/env python3

import glob
import multiprocessing
import os
import shutil
import sys

import cv2
import imageio
import numpy as np
from PIL import Image


def main():
    if 'DATA_ROOT' not in os.environ:
        print('Set the DATA_ROOT environment variable to the parent dir of the '
              'inria_holidays directory.')
        sys.exit(1)

    pool = multiprocessing.Pool()
    data_root = os.environ['DATA_ROOT']
    image_paths = glob.glob(f'{data_root}/inria_holidays/jpg/*')
    os.makedirs(f'{data_root}/inria_holidays/jpg_small')
    pool.map(rotate_resize_save, image_paths)
    shutil.rmtree(f'{data_root}/inria_holidays/jpg')


def rotate_resize_save(src_path):
    print('Processing', src_path)
    dst_path = src_path.replace('holidays/jpg', 'holidays/jpg_small')
    im = load_image_with_proper_orientation(src_path)
    im = crop_center_square(im)
    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
    imageio.imwrite(dst_path, im, quality=95)


def load_image_with_proper_orientation(filepath):
    try:
        image = Image.open(filepath)
        orientation_exif_index = 274
        exif_info = dict(image._getexif().items())

        if exif_info[orientation_exif_index] == 3:
            image = image.rotate(180, expand=True)
        elif exif_info[orientation_exif_index] == 6:
            image = image.rotate(270, expand=True)
        elif exif_info[orientation_exif_index] == 8:
            image = image.rotate(90, expand=True)
        return np.asarray(image)
    except (AttributeError, KeyError, IndexError):
        # No EXIF found
        return imageio.imread(filepath)


def crop_center_square(im):
    height, width, channels = im.shape
    if height == width:
        return im
    elif height < width:
        x_start = (width - height) // 2
        im = im[:, x_start:x_start + height, :]
    else:
        y_start = (height - width) // 2
        im = im[y_start:y_start + width, :, :]
    return im


if __name__ == '__main__':
    main()
