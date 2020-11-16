#!/usr/bin/env python3

import os
import glob
import sys
import re


def main():
    if 'DATA_ROOT' not in os.environ:
        print(
            'Set the DATA_ROOT environment variable to the parent dir of the 3dhp '
            'directory.')
        sys.exit(1)

    data_root = os.environ['DATA_ROOT']
    filepaths = sorted(glob.glob(f'{data_root}/3dhp/S*/**/imageSequence/*.jpg', recursive=True))
    for path in filepaths:
        i_frame = int(re.search(r'_(\d+).jpg$', path)[1])
        if i_frame % 5 == 0:
            print(path)


if __name__ == '__main__':
    main()
