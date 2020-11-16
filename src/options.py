import argparse
import logging
import os
import socket
import sys

import paths
import util

FLAGS = argparse.Namespace()


def initialize(parser):
    parser.add_argument('--loglevel', type=str, default='error')
    parser.parse_args(namespace=FLAGS)
    loglevel = dict(error=40, warning=30, info=20, debug=10)[FLAGS.loglevel]
    simple_formatter = logging.Formatter('{asctime}-{levelname:^1.1} -- {message}', style='{')
    print_handler = logging.StreamHandler(sys.stdout)
    print_handler.setLevel(loglevel)
    print_handler.setFormatter(simple_formatter)
    logging.basicConfig(level=loglevel, handlers=[print_handler])


def initialize_with_logfiles(parser):
    parser.add_argument('--logdir', type=str, default='default_logdir')
    parser.add_argument('--file', type=open, action=ParseFromFileAction)
    parser.add_argument('--loglevel', type=str, default='info')
    parser.parse_args(namespace=FLAGS)
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
    handlers = [simple_logfile_handler, detailed_logfile_handler]

    if sys.stdout.isatty():
        # We only print the log messages to stdout if it's a terminal (tty).
        # Otherwise it goes to the log file.
        print_handler = logging.StreamHandler(sys.stdout)
        print_handler.setLevel(loglevel)
        print_handler.setFormatter(simple_formatter)
        handlers.append(print_handler)
    else:
        # Since we don't want to print the log to stdout, we also redirect stderr to the logfile to
        # save errors for future inspection. But stdout is still stdout.
        sys.stderr.flush()
        new_err_file = open(detailed_logfile_path, 'ab+', 0)
        STDERR_FILENO = 2
        os.dup2(new_err_file.fileno(), STDERR_FILENO)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers)


class ParseFromFileAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            lines = f.read().splitlines()
            args = [f'--{line}' for line in lines if line and not line.startswith('#')]
            parser.parse_args(args, namespace)


class HyphenToUnderscoreAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.replace('-', '_'))


class YesNoAction(argparse.Action):
    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        positive_opts = option_strings
        if not all(opt.startswith('--') for opt in positive_opts):
            raise ValueError('Yes/No arguments must be prefixed with --')
        if any(opt.startswith('--no-') for opt in positive_opts):
            raise ValueError(
                'Yes/No arguments cannot start with --no-, the --no- version will be '
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
