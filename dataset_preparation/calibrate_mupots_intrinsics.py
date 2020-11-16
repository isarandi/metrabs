#!/usr/bin/env python3
import json
import os
import sys

import numpy as np
import scipy.io


def main():
    if 'DATA_ROOT' not in os.environ:
        print('Set the DATA_ROOT environment variable to the parent dir of the mupots directory.')
        sys.exit(1)
    intrinsics_per_sequence = {}
    for i_seq in range(1, 21):
        anno_path = f'{os.environ["DATA_ROOT"]}/mupots/TS{i_seq}/annot.mat'
        anno = scipy.io.loadmat(anno_path, struct_as_record=False, squeeze_me=True)['annotations']
        points2d = np.concatenate([x.annot2.T for x in np.nditer(anno) if x.isValidFrame])
        points3d = np.concatenate([x.annot3.T for x in np.nditer(anno) if x.isValidFrame])
        intrinsics_per_sequence[f'TS{i_seq}'] = estimate_intrinsic_matrix(points2d, points3d)

    with open(f'{os.environ["DATA_ROOT"]}/mupots/camera_intrinsics.json', 'w') as file:
        return json.dump(intrinsics_per_sequence, file)


def estimate_intrinsic_matrix(points2d, points3d):
    n_rows = len(points2d) * 2
    A = np.empty((n_rows, 4))
    b = np.empty((n_rows, 1))
    for i, ((x2, y2), (x3, y3, z3)) in enumerate(zip(points2d, points3d)):
        A[2 * i] = [x3 / z3, 0, 1, 0]
        A[2 * i + 1] = [0, y3 / z3, 0, 1]
        b[2 * i] = [x2]
        b[2 * i + 1] = [y2]
    fx, fy, cx, cy = np.linalg.lstsq(A, b, rcond=None)[0][:, 0]
    return [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]


if __name__ == '__main__':
    main()
