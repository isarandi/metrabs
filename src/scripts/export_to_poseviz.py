#!/usr/bin/env python3

import argparse
import collections
import logging

import numpy as np

import options
import util
from options import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, default=None)
    options.initialize(parser)

    if FLAGS.out_path is None:
        FLAGS.out_path = FLAGS.in_path.replace('.npz', '.json')

    logging.debug(f'Loading from {FLAGS.in_path}')
    a = np.load(FLAGS.in_path, allow_pickle=True)
    all_results_3d = collections.defaultdict(list)
    for image_path, coords3d_pred in zip(a['image_path'], a['coords3d_pred_world']):
        all_results_3d[image_path.decode('utf8')].append(coords3d_pred.tolist())
    logging.info(f'Writing to file {FLAGS.out_path}')
    util.dump_json(all_results_3d, FLAGS.out_path)


if __name__ == '__main__':
    main()
