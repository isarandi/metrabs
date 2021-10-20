import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'
os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = 'true'
# Must import cv2 early
# noinspection PyUnresolvedReferences
import cv2

# Must import TF early due to problem with PyCharm debugger otherwise.
# noinspection PyUnresolvedReferences
import tensorflow as tf
import argparse
from options import logger
import os
import shlex
import socket
import sys
import paths
import options
import tfu
from options import FLAGS
import matplotlib.pyplot as plt
import util


def initialize(args=None):
    options.initialize_with_logfiles(get_parser(), args)
    logger.info(f'-- Starting --')
    logger.info(f'Host: {socket.gethostname()}')
    logger.info(f'Process id (pid): {os.getpid()}')

    if FLAGS.comment:
        logger.info(f'Comment: {FLAGS.comment}')
    logger.info(f'Raw command: {" ".join(map(shlex.quote, sys.argv))}')
    logger.info(f'Parsed flags: {FLAGS}')
    tfu.set_data_format(FLAGS.data_format)
    tfu.set_dtype(tf.float32 if FLAGS.dtype == 'float32' else tf.float16)

    if FLAGS.batch_size_test is None:
        FLAGS.batch_size_test = FLAGS.batch_size

    if FLAGS.checkpoint_dir is None:
        FLAGS.checkpoint_dir = FLAGS.logdir

    FLAGS.checkpoint_dir = util.ensure_absolute_path(
        FLAGS.checkpoint_dir, root=f'{paths.DATA_ROOT}/experiments')
    os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)

    if not FLAGS.pred_path:
        FLAGS.pred_path = f'predictions_{FLAGS.dataset}.npz'
    base = os.path.dirname(FLAGS.load_path) if FLAGS.load_path else FLAGS.checkpoint_dir
    FLAGS.pred_path = util.ensure_absolute_path(FLAGS.pred_path, base)

    if FLAGS.bone_length_dataset is None:
        FLAGS.bone_length_dataset = FLAGS.dataset

    if FLAGS.model_joints is None:
        FLAGS.model_joints = FLAGS.dataset

    if FLAGS.output_joints is None:
        FLAGS.output_joints = FLAGS.dataset

    if FLAGS.load_path:
        if FLAGS.load_path.endswith('.index') or FLAGS.load_path.endswith('.meta'):
            FLAGS.load_path = os.path.splitext(FLAGS.load_path)[0]
        FLAGS.load_path = util.ensure_absolute_path(FLAGS.load_path, FLAGS.checkpoint_dir)

    tf.random.set_seed(FLAGS.seed)
    if FLAGS.viz:
        plt.switch_backend('TkAgg')

    FLAGS.backbone = FLAGS.backbone.replace('_', '-')

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    if FLAGS.dtype == 'float16':
        tf.keras.mixed_precision.set_global_policy('mixed_float16')


def get_parser():
    parser = argparse.ArgumentParser(
        description='MeTRAbs 3D Human Pose Estimator', allow_abbrev=False)
    # Essentials
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers to run. Default is min(12, num_cpus)')

    # Task options (what to do)
    parser.add_argument('--train', action=options.BoolAction, help='Train the model.')
    parser.add_argument('--predict', action=options.BoolAction, help='Test the model.')
    parser.add_argument('--export-file', type=str, help='Export filename.')
    parser.add_argument('--export-smpl', action=options.BoolAction)

    # Monitoring options
    parser.add_argument('--viz', action=options.BoolAction,
                        help='Create graphical user interface for visualization.')

    # Loading and input processing options
    parser.add_argument('--load-path', type=str, default=None,
                        help='Path of model checkpoint to load in the beginning.')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory path of model checkpoints.')
    parser.add_argument('--init', type=str, default='pretrained',
                        help='How to initialize the weights: "scratch" or "pretrained".')
    parser.add_argument('--init-path', type=str, default=None,
                        help="""Path of the pretrained checkpoint to initialize from once
                        at the very start of training (i.e. not when resuming!).
                        To restore for resuming an existing training use the --load-path option.""")
    parser.add_argument('--proc-side', type=int, default=256,
                        help='Side length of image as processed by network.')
    parser.add_argument('--geom-aug', action=options.BoolAction, default=True,
                        help='Training data augmentations such as rotation, scaling, translation '
                             'etc.')
    parser.add_argument('--test-aug', action=options.BoolAction,
                        help='Apply augmentations to test images.')
    parser.add_argument('--rot-aug', type=float,
                        help='Rotation augmentation in degrees.', default=20)
    parser.add_argument('--scale-aug-up', type=float,
                        help='Scale augmentation in percent.', default=25)
    parser.add_argument('--scale-aug-down', type=float,
                        help='Scale augmentation in percent.', default=25)
    parser.add_argument('--shift-aug', type=float,
                        help='Shift augmentation in percent.', default=10)
    parser.add_argument('--test-subjects', type=str, default=None,
                        help='Test subjects.')
    parser.add_argument('--valid-subjects', type=str, default=None,
                        help='Validation subjects.')
    parser.add_argument('--train-subjects', type=str, default=None,
                        help='Training subjects.')

    parser.add_argument('--train-on', type=str, default='train',
                        help='Training part.')
    parser.add_argument('--validate-on', type=str, default='valid',
                        help='Validation part.')
    parser.add_argument('--test-on', type=str, default='test',
                        help='Test part.')

    # Training options
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32'],
                        help='The floating point type to use for computations.')
    parser.add_argument('--validate-period', type=int, default=None,
                        help='Periodically validate during training, every this many steps.')

    # Optimizer options
    parser.add_argument('--weight-decay', type=float, default=3e-3)
    parser.add_argument('--base-learning-rate', type=float, default=1e-4,
                        help='Learning rate of the optimizer.')

    # Test options
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-size-test', type=int, default=110)
    parser.add_argument('--multiepoch-test', action=options.BoolAction)

    parser.add_argument('--data-format', type=str, default='NHWC',
                        help='Data format used internally.')

    parser.add_argument('--stride-train', type=int, default=32)
    parser.add_argument('--stride-test', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--comment', type=str, default=None)

    parser.add_argument('--dataset2d', type=str, default='mpii',
                        action=options.HyphenToUnderscoreAction)

    parser.add_argument('--dataset', type=str, default='h36m',
                        action=options.HyphenToUnderscoreAction)
    parser.add_argument('--model-joints', type=str, default=None,
                        action=options.HyphenToUnderscoreAction)
    parser.add_argument('--output-joints', type=str, default=None,
                        action=options.HyphenToUnderscoreAction)
    parser.add_argument('--backbone', type=str, default='resnet50V2',
                        help='Backbone of the predictor network.')
    parser.add_argument('--pred-path', type=str, default=None)
    parser.add_argument('--depth', type=int, default=8,
                        help='Number of voxels along the z axis for volumetric prediction')
    parser.add_argument('--batch-size-2d', type=int, default=32)

    parser.add_argument('--centered-stride', action=options.BoolAction, default=True)
    parser.add_argument('--box-size-mm', type=float, default=2200)
    parser.add_argument('--universal-skeleton', action=options.BoolAction)

    parser.add_argument('--weak-perspective', action=options.BoolAction)
    parser.add_argument('--mean-relative', action=options.BoolAction)

    parser.add_argument('--loss2d-factor', type=float, default=0.1)
    parser.add_argument('--absloss-factor', type=float, default=1.0)
    parser.add_argument('--tdhp-to-mpii-shift-factor', type=float, default=0.2)

    parser.add_argument('--bone-length-dataset', type=str)

    parser.add_argument('--partial-visibility-prob', type=float, default=0)
    parser.add_argument('--occlude-aug-prob', type=float, default=0.7)
    parser.add_argument('--occlude-aug-prob-2d', type=float, default=0.7)
    parser.add_argument('--occlude-aug-scale', type=float, default=0.8)
    parser.add_argument('--background-aug-prob', type=float, default=0)
    parser.add_argument('--color-aug', action=options.BoolAction, default=True)
    parser.add_argument('--compatibility-mode', action=options.BoolAction)
    parser.add_argument('--antialias-train', type=int, default=1)
    parser.add_argument(
        '--antialias-test', type=int, default=1)  # 4 can be more accurate
    parser.add_argument(
        '--image-interpolation-train', type=str, default='linear')  # 'nearest' can be faster
    parser.add_argument('--image-interpolation-test', type=str, default='linear')
    parser.add_argument('--transform-coords', action=options.BoolAction)

    parser.add_argument('--ghost-bn', type=str, default='')
    parser.add_argument('--group-norm', action=options.BoolAction)
    parser.add_argument('--shuffle-batch', action=options.BoolAction)
    parser.add_argument('--finetune-in-inference-mode', type=int, default=0)
    parser.add_argument('--scale-agnostic-loss', action=options.BoolAction)
    parser.add_argument('--training-steps', type=int)
    parser.add_argument('--multi-gpu', action=options.BoolAction)
    parser.add_argument('--checkpoint-period', type=int, default=2000)
    parser.add_argument('--model-class', type=str, default='Metrabs')
    return parser
