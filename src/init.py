import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Must import cv2 early
# noinspection PyUnresolvedReferences
import cv2

# Must import TF and slim early due to problem with PyCharm debugger otherwise.
# noinspection PyUnresolvedReferences
import tensorflow as tf
import tf_slim as slim
import argparse
import logging
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


def initialize():
    tf.compat.v1.disable_eager_execution()

    options.initialize_with_logfiles(get_parser())
    logging.info(f'-- Starting --')
    logging.info(f'Host: {socket.gethostname()}')
    logging.info(f'Process id (pid): {os.getpid()}')

    if FLAGS.comment:
        logging.info(f'Comment: {FLAGS.comment}')
    logging.info(f'Raw command: {" ".join(map(shlex.quote, sys.argv))}')
    logging.info(f'Parsed flags: {FLAGS}')
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

    if FLAGS.load_path:
        if FLAGS.load_path.endswith('.index') or FLAGS.load_path.endswith('.meta'):
            FLAGS.load_path = os.path.splitext(FLAGS.load_path)[0]
        FLAGS.load_path = util.ensure_absolute_path(FLAGS.load_path, FLAGS.checkpoint_dir)

    # Override the default data format in slim layers
    enter_context(slim.arg_scope(
        [slim.conv2d, slim.conv3d, slim.conv3d_transpose, slim.conv2d_transpose, slim.avg_pool2d,
         slim.separable_conv2d, slim.max_pool2d, slim.batch_norm, slim.spatial_softmax],
        data_format=tfu.data_format()))

    # Override default paddings to SAME
    enter_context(slim.arg_scope([slim.avg_pool2d, slim.max_pool2d], padding='SAME'))
    tf.compat.v2.random.set_seed(FLAGS.seed)
    if FLAGS.gui:
        plt.switch_backend('TkAgg')


def get_parser():
    parser = argparse.ArgumentParser(description='MeTRo-Pose3D', allow_abbrev=False)
    # Essentials
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers to run.')

    # Task options (what to do)
    parser.add_argument('--train', action=options.YesNoAction, help='Train the model.')
    parser.add_argument('--test', action=options.YesNoAction, help='Test the model.')
    parser.add_argument('--export-file', type=str, help='Export filename.')
    parser.add_argument('--export-smpl', action=options.YesNoAction)

    # Monitoring options
    parser.add_argument('--gui', action=options.YesNoAction,
                        help='Create graphical user interface for visualization.')
    parser.add_argument('--hook-seconds', type=float, default=15,
                        help='How often to call log, imshow and summary hooks.')
    parser.add_argument('--tensorboard', action=options.YesNoAction, default=True,
                        help='Apply augmentations to test images.')

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
    parser.add_argument('--geom-aug', action=options.YesNoAction, default=True,
                        help='Training data augmentations such as rotation, scaling, translation '
                             'etc.')
    parser.add_argument('--test-aug', action=options.YesNoAction,
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
    parser.add_argument('--epochs', type=float, default=0,
                        help='Number of training epochs, 0 means unlimited.')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32'],
                        help='The floating point type to use for computations.')
    parser.add_argument('--validate-period', type=float, default=None,
                        help='Periodically validate during training, every this many epochs.')

    # Optimizer options
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--weight-decay', type=float, default=3e-3)
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate of the optimizer.')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='Epsilon for the Adam optimizer (called epsilon-hat in the paper).')

    # Test options
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-size-test', type=int, default=110)
    parser.add_argument('--multiepoch-test', action=options.YesNoAction)
    parser.add_argument('--data-format', type=str, default='NHWC',
                        help='Data format used internally.')

    parser.add_argument('--stride-train', type=int, default=32)
    parser.add_argument('--stride-test', type=int, default=4)

    parser.add_argument('--max-unconsumed', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--comment', type=str, default=None)

    parser.add_argument('--dataset2d', type=str, default='mpii',
                        action=options.HyphenToUnderscoreAction)

    parser.add_argument('--dataset', type=str, default='h36m',
                        action=options.HyphenToUnderscoreAction)
    parser.add_argument('--architecture', type=str, default='resnet_v2_50',
                        action=options.HyphenToUnderscoreAction,
                        help='Architecture of the predictor network.')
    parser.add_argument('--pred-path', type=str, default=None)
    parser.add_argument('--depth', type=int, default=8,
                        help='Number of voxels along the z axis for volumetric prediction')
    parser.add_argument('--train-mixed', action=options.YesNoAction, default=True)
    parser.add_argument('--batch-size-2d', type=int, default=32)

    parser.add_argument('--centered-stride', action=options.YesNoAction, default=True)
    parser.add_argument('--box-size-mm', type=float, default=2200)
    parser.add_argument('--universal-skeleton', action=options.YesNoAction)

    parser.add_argument('--partial-visibility', action=options.YesNoAction)
    parser.add_argument('--metrabs-plus', action=options.YesNoAction)
    parser.add_argument('--init-logits-random', action=options.YesNoAction, default=True)
    parser.add_argument('--weak-perspective', action=options.YesNoAction, default=True)
    parser.add_argument('--mean-relative', action=options.YesNoAction)

    parser.add_argument('--loss2d-factor', type=float, default=0.1)
    parser.add_argument('--absloss-factor', type=float, default=1.0)
    parser.add_argument('--tdhp-to-mpii-shift-factor', type=float, default=0.2)
    parser.add_argument('--scale-recovery', type=str, default='metro')

    parser.add_argument('--bone-length-dataset', type=str)
    parser.add_argument('--batchnorm-together-2d3d', action=options.YesNoAction)

    parser.add_argument('--occlude-aug-prob', type=float, default=0.7)
    parser.add_argument('--occlude-aug-scale', type=float, default=0.8)
    parser.add_argument('--background-aug-prob', type=float, default=0)
    parser.add_argument('--color-aug', action=options.YesNoAction, default=True)
    parser.add_argument('--compatibility-mode', action=options.YesNoAction)
    parser.add_argument('--antialias-train', type=int, default=1)
    parser.add_argument('--antialias-test', type=int, default=4)
    parser.add_argument('--image-interpolation-train', type=str, default='nearest')
    parser.add_argument('--image-interpolation-test', type=str, default='linear')

    return parser


_contexts = []


def enter_context(context):
    """Enter a context and keep a reference to it alive."""
    # This fixes the following issue: When `__enter__`ing a local arg_scope variable in a function,
    # the arg_scope seemingly exits when the function returns.
    # This happens because arg_scope is implemented using contextlib.context_manager, so this
    # context manager wraps a generator.
    # However, when the GC cleans up a generator object that hasn't finished iterating yet,
    # it calls the generator's close() method which results in a GeneratorExit exception inside the
    # generator. This can trigger some exception handler or finally block in the generator,
    # which in case of arg_scope basically does the same as an __exit__ (it pops the scope from the
    #  stack)
    context.__enter__()
    global _contexts
    _contexts.append(context)
