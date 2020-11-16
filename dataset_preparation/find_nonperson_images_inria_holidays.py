#!/usr/bin/env python3
import os
import pickle
import sys


def main():
    if 'DATA_ROOT' not in os.environ:
        print(
            'Set the DATA_ROOT environment variable to the parent dir of the inria_holidays '
            'directory.')
        sys.exit(1)

    data_root = os.environ['DATA_ROOT']
    with open(f'{data_root}/inria_holidays/yolov3_person_detections.pkl', 'rb') as f:
        detections_all = pickle.load(f)

    filenames_without_detection = sorted([
        filename for filename, detections in detections_all.items() if not detections])

    with open(f'{data_root}/inria_holidays/non_person_images.txt', 'w') as f:
        f.write('\n'.join(filenames_without_detection))


if __name__ == '__main__':
    main()
