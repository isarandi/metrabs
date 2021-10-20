import argparse
import logging
import os
import shlex
import socket
import sys

import paths
import util

FLAGS = argparse.Namespace()
logger = logging.getLogger('metrabs')


def initialize(parser, args=None):
    parser.add_argument('--loglevel', type=str, default='error')
    parser.parse_args(args=args, namespace=FLAGS)
    loglevel = dict(error=40, warning=30, info=20, debug=10)[FLAGS.loglevel]
    simple_formatter = logging.Formatter('{asctime}-{levelname:^1.1} -- {message}', style='{')

    if sys.stdout.isatty():
        # Make sure that the log messages appear above the tqdm progess bars
        import tqdm
        class TQDMFile:
            def write(self, x):
                if len(x.rstrip()) > 0:
                    tqdm.tqdm.write(x, file=sys.stdout)

        out_stream = TQDMFile()
    else:
        out_stream = sys.stdout

    print_handler = logging.StreamHandler(out_stream)
    print_handler.setLevel(loglevel)
    print_handler.setFormatter(simple_formatter)
    logger.setLevel(loglevel)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.addHandler(print_handler)


def initialize_with_logfiles(parser, args=None):
    parser.add_argument('--logdir', type=str, default='default_logdir')
    parser.add_argument('--file', type=open, action=ParseFromFileAction)
    parser.add_argument('--loglevel', type=str, default='info')
    if isinstance(args, str):
        args = shlex.split(args)

    parser.parse_args(args=args, namespace=FLAGS)
    loglevel = dict(error=40, warning=30, info=20, debug=10)[FLAGS.loglevel]
    FLAGS.logdir = util.ensure_absolute_path(FLAGS.logdir, root=f'{paths.DATA_ROOT}/experiments')
    os.makedirs(FLAGS.logdir, exist_ok=True)
    simple_logfile_path = f'{FLAGS.logdir}/log.txt'
    detailed_logfile_path = f'{FLAGS.logdir}/log_detailed.txt'
    simple_logfile_handler = logging.FileHandler(simple_logfile_path)
    simple_logfile_handler.setLevel(loglevel)
    detailed_logfile_handler = logging.FileHandler(detailed_logfile_path)

    simple_formatter = logging.Formatter('{asctime}-{levelname:^1.1} -- {message}', style='{')
    hostname = socket.gethostname().split('.', 1)[0]
    detailed_formatter = logging.Formatter(
        f'{{asctime}} - {hostname} - {{process}} - {{processName:^12.12}} -' +
        ' {threadName:^12.12} - {name:^12.12} - {levelname:^7.7} -- {message}', style='{')

    simple_logfile_handler.setFormatter(simple_formatter)
    detailed_logfile_handler.setFormatter(detailed_formatter)
    logger.addHandler(simple_logfile_handler)
    logger.addHandler(detailed_logfile_handler)

    if sys.stdout.isatty():
        # We only print the log messages to stdout if it's a terminal (tty).
        # Otherwise it goes to the log file.

        # Make sure that the log messages appear above the tqdm progess bars
        import tqdm
        class TQDMFile:
            def write(self, x):
                if len(x.rstrip()) > 0:
                    tqdm.tqdm.write(x, file=sys.stdout)

        print_handler = logging.StreamHandler(TQDMFile())
        print_handler.setLevel(loglevel)
        print_handler.setFormatter(simple_formatter)
        logger.addHandler(print_handler)
    else:
        # Since we don't want to print the log to stdout, we also redirect stderr to the logfile to
        # save errors for future inspection. But stdout is still stdout.
        sys.stderr.flush()
        new_err_file = open(detailed_logfile_path, 'ab+', 0)
        STDERR_FILENO = 2
        os.dup2(new_err_file.fileno(), STDERR_FILENO)

    logger.setLevel(logging.DEBUG)


class ParseFromFileAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            lines = f.read().splitlines()
            args = [f'--{line}' for line in lines if line and not line.startswith('#')]
            parser.parse_args(args, namespace)


class HyphenToUnderscoreAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.replace('-', '_'))


class BoolAction(argparse.Action):
    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        positive_opts = option_strings
        if not all(opt.startswith('--') for opt in positive_opts):
            raise ValueError('Boolean arguments must be prefixed with --')
        if any(opt.startswith('--no-') for opt in positive_opts):
            raise ValueError(
                'Boolean arguments cannot start with --no-, the --no- version will be '
                'auto-generated')

        negative_opts = ['--no-' + opt[2:] for opt in positive_opts]
        opts = [*positive_opts, *negative_opts]
        super().__init__(
            opts, dest, nargs=0, const=None, default=default, required=required, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string.startswith('--no-'):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)
