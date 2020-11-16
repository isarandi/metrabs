#!/usr/bin/env python3
import glob
import multiprocessing
import os
import pathlib
import sys

import imageio


def main():
    if 'DATA_ROOT' not in os.environ:
        print('Set the DATA_ROOT environment variable to the parent dir of the 3dhp directory.')
        sys.exit(1)

    data_root = os.environ['DATA_ROOT']
    every_nth = int(sys.argv[1])

    pool = multiprocessing.Pool()
    video_paths = glob.glob(f'{data_root}/3dhp/**/video_[01245678].avi', recursive=True)
    args = [(p, every_nth) for p in video_paths]
    pool.starmap(extract_frames, args)


def extract_frames(src_video_path, every_nth):
    print('Processing', src_video_path)
    video_name = pathlib.Path(src_video_path).stem
    i_video = int(video_name.split('_')[1])
    dst_folder_path = os.path.dirname(src_video_path)
    with imageio.get_reader(src_video_path, 'ffmpeg') as reader:
        for i_frame, frame in enumerate(reader):
            if i_frame % every_nth == 0:
                dst_path = f'{dst_folder_path}/img_{i_video + 1}_{i_frame:06d}.jpg'
                imageio.imwrite(dst_path, frame, quality=95)


if __name__ == '__main__':
    main()
