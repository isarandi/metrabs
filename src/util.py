import contextlib
import ctypes
import datetime
import functools
import hashlib
import inspect
import itertools
import json
import logging
import multiprocessing as mp
import os
import os.path
import pickle
import signal
import threading
import traceback

import numpy as np

import paths

TRAIN = 0
VALID = 1
TEST = 2


def cache_result_on_disk(path, forced=None, min_time=None):
    """Helps with caching and restoring the results of a function call on disk.
    Specifically, it returns a function decorator that makes a function cache its result in a file.
    It only evaluates the function once, to generate the cached file. The decorator also adds a
    new keyword argument to the function, called 'forced_cache_update' that can explicitly force
    regeneration of the cached file.

    It has rudimentary handling of arguments by hashing their json representation and appending it
    the hash to the cache filename. This somewhat limited, but is enough for the current uses.

    Set `min_time` to the last significant change to the code within the function.
    If the cached file is older than this `min_time`, the file is regenerated.

    Usage:
        @cache_result_on_disk('/some/path/to/a/file', min_time='2025-12-27T10:12:32')
        def some_function(some_arg):
            ....
            return stuff

    Args:
        path: The path where the function's result is stored.
        forced: do not load from disk, always recreate the cached version
        min_time: recreate cached file if its modification timestamp (mtime) is older than this
           param. The format is like 2025-12-27T10:12:32 (%Y-%m-%dT%H:%M:%S)

    Returns:
        The decorator.
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            inner_forced = forced if forced is not None else kwargs.get('forced_cache_update')
            if 'forced_cache_update' in kwargs:
                del kwargs['forced_cache_update']

            bound_args = inspect.signature(f).bind(*args, **kwargs)
            args_json = json.dumps((bound_args.args, bound_args.kwargs), sort_keys=True)
            hash_string = hashlib.sha1(str(args_json).encode('utf-8')).hexdigest()[:12]

            if args or kwargs:
                noext, ext = os.path.splitext(path)
                suffixed_path = f'{noext}_{hash_string}{ext}'
            else:
                suffixed_path = path

            if not inner_forced and is_file_newer(suffixed_path, min_time):
                logging.debug(f'Loading cached data from {suffixed_path}')
                try:
                    return load_pickle(suffixed_path)
                except Exception as e:
                    print(str(e))
                    logging.error(f'Could not load from {suffixed_path}')
                    raise e

            if os.path.exists(suffixed_path):
                logging.debug(f'Recomputing data for {suffixed_path}')
            else:
                logging.debug(f'Computing data for {suffixed_path}')

            result = f(*args, **kwargs)
            dump_pickle(result, suffixed_path)

            if args or kwargs:
                write_file(args_json, f'{os.path.dirname(path)}/hash_{hash_string}')

            return result

        return wrapped

    return decorator


def timestamp(simplified=False):
    stamp = datetime.datetime.now().isoformat()
    if simplified:
        return stamp.replace(':', '-').replace('.', '-')
    return stamp


class FormattableArray:
    def __init__(self, array):
        self.array = np.asarray(array)

    def __format__(self, format_spec):
        # with np.printoptions(
        with numpy_printoptions(
                formatter={'float': lambda x: format(x, format_spec)},
                linewidth=10 ** 6, threshold=10 ** 6):
            return str(self.array)


formattable_array = FormattableArray


@contextlib.contextmanager
def numpy_printoptions(*args, **kwargs):
    original_printoptions = np.get_printoptions()
    try:
        np.set_printoptions(*args, **kwargs)
        yield
    finally:
        np.set_printoptions(**original_printoptions)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(data, file_path, protocol=pickle.HIGHEST_PROTOCOL):
    ensure_path_exists(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol)


def dump_json(data, path):
    ensure_path_exists(path)
    with open(path, 'w') as file:
        return json.dump(data, file)


def write_file(content, path, is_binary=False):
    mode = 'wb' if is_binary else 'w'
    ensure_path_exists(path)
    with open(path, mode) as f:
        if not is_binary:
            content = str(content)
        f.write(content)
        f.flush()


def ensure_path_exists(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def read_file(path, is_binary=False):
    mode = 'rb' if is_binary else 'r'
    with open(path, mode) as f:
        return f.read()


def split_path(path):
    return os.path.normpath(path).split(os.path.sep)


def last_path_components(path, n_components):
    components = split_path(path)
    return os.path.sep.join(components[-n_components:])


def index_of_first_true(seq, default=None):
    return next((i for i, x in enumerate(seq) if x), default)


def plot_mean_std(ax, x, ys, axis=0):
    mean = np.mean(ys, axis=axis)
    std = np.std(ys, axis=axis)

    ax.plot(x, mean)
    ax.fill_between(x, mean - std, mean + std, alpha=0.3)


def iterate_repeatedly(seq, shuffle_before_each_epoch=False, rng=None):
    """Iterates over and yields the elements of `iterable` `n_epoch` times.
    if `shuffle_before_each_epoch` is True, the elements are put in a list and shuffled before
    every pass over the data, including the first."""

    if rng is None:
        rng = np.random.RandomState()

    # create a (shallow) copy so shuffling only applies to the copy.
    seq = list(seq)
    for i_epoch in itertools.count():
        logging.debug(f'starting epoch {i_epoch}')
        if shuffle_before_each_epoch:
            logging.debug(f'shuffling {i_epoch}')
            rng.shuffle(seq)
        yield from seq
        logging.debug(f'ended epoch {i_epoch}')


def random_partial_box(random_state):
    def generate():
        x1 = random_state.uniform(0, 0.5)
        x2, y2 = random_state.uniform(0.5, 1, size=2)
        side = x2 - x1
        if not 0.5 < side < y2:
            return None
        return np.array([x1, y2 - side, side, side])

    while True:
        box = generate()
        if box is not None:
            return box


def random_partial_subbox(box, random_state):
    subbox = random_partial_box(random_state)
    topleft = box[:2] + subbox[:2] * box[2:]
    size = subbox[2:] * box[2:]
    return np.concatenate([topleft, size])


def init_worker_process():
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

    terminate_on_parent_death()
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def new_rng(rng):
    if rng is not None:
        return np.random.RandomState(rng.randint(2 ** 32))
    else:
        return np.random.RandomState()


def advance_rng(rng, n_generated_ints):
    for _ in range(n_generated_ints):
        rng.randint(2)


def choice(items, rng):
    return items[rng.randint(len(items))]


def random_uniform_disc(rng):
    """Samples a random 2D point from the unit disc with a uniform distribution."""
    angle = rng.uniform(-np.pi, np.pi)
    radius = np.sqrt(rng.uniform(0, 1))
    return radius * np.array([np.cos(angle), np.sin(angle)])


def init_worker_process_flags(flags):
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['KMP_INIT_AT_FORK'] = 'FALSE'
    from options import FLAGS
    for key in flags.__dict__:
        setattr(FLAGS, key, getattr(flags, key))
    import tfu
    tfu.set_data_format(FLAGS.data_format)
    init_worker_process()


def terminate_on_parent_death():
    prctl = ctypes.CDLL("libc.so.6").prctl
    PR_SET_PDEATHSIG = 1
    prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


def safe_subprocess_main_with_flags(flags, func, *args, **kwargs):
    if flags.gui:
        import matplotlib.pyplot as plt
        plt.switch_backend('TkAgg')
    init_worker_process_flags(flags)
    return func(*args, **kwargs)


def is_file_newer(path, min_time=None):
    if min_time is None:
        return os.path.exists(path)
    min_time = datetime.datetime.strptime(min_time, '%Y-%m-%dT%H:%M:%S').timestamp()
    return os.path.exists(path) and os.path.getmtime(path) >= min_time


def safe_fun(f, args):
    try:
        return f(*args)
    except BaseException:
        traceback.print_exc()
        raise


def init_cuda():
    import cv2
    cv2.cuda.resetDevice()
    print(cv2.cuda.setDevice(0))


class BoundedPool:
    """Wrapper around multiprocessing.Pool that blocks on task submission (`apply_async`) if
    there are already `task_buffer_size` tasks under processing. This can be useful in
    throttling the task producer thread and avoiding too many tasks piling up in the queue and
    eating up too much RAM."""

    def __init__(self, n_processes, task_buffer_size):
        self.pool = mp.Pool(processes=n_processes)
        self.task_semaphore = threading.Semaphore(task_buffer_size)

    def apply_async(self, f, args, callback=None):
        self.task_semaphore.acquire()

        def on_task_completion(result):
            if callback is not None:
                callback(result)
            self.task_semaphore.release()

        self.pool.apply_async(safe_fun, args=(f, args), callback=on_task_completion)

    def close(self):
        self.pool.close()

    def join(self):
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.close()
        self.pool.join()
        self.pool.terminate()


def all_disjoint(*seqs):
    union = set()
    for item in itertools.chain(*seqs):
        if item in union:
            return False
        union.add(item)
    return True


def is_running_in_jupyter_notebook():
    try:
        # noinspection PyUnresolvedReferences
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def progressbar(*args, **kwargs):
    import tqdm.notebook
    import sys
    if is_running_in_jupyter_notebook():
        return tqdm.notebook.tqdm(*args, **kwargs)
    elif sys.stdout.isatty():
        return tqdm.tqdm(*args, dynamic_ncols=True, **kwargs)
    else:
        return args[0]


def ensure_absolute_path(path, root=paths.DATA_ROOT):
    if not root:
        return path

    if os.path.isabs(path):
        return path
    else:
        return os.path.join(root, path)


def invert_permutation(permutation):
    return np.arange(len(permutation))[np.argsort(permutation)]


def load_json(path):
    with open(path) as file:
        return json.load(file)


def cycle_over_colors(range_zero_one=False):
    """Returns a generator that cycles over a list of nice colors, indefinitely."""
    colors = ((0.12156862745098039, 0.46666666666666667, 0.70588235294117652),
              (1.0, 0.49803921568627452, 0.054901960784313725),
              (0.17254901960784313, 0.62745098039215685, 0.17254901960784313),
              (0.83921568627450982, 0.15294117647058825, 0.15686274509803921),
              (0.58039215686274515, 0.40392156862745099, 0.74117647058823533),
              (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
              (0.8901960784313725, 0.46666666666666667, 0.76078431372549016),
              (0.49803921568627452, 0.49803921568627452, 0.49803921568627452),
              (0.73725490196078436, 0.74117647058823533, 0.13333333333333333),
              (0.090196078431372548, 0.74509803921568629, 0.81176470588235294))

    if not range_zero_one:
        colors = [[c * 255 for c in color] for color in colors]

    return itertools.cycle(colors)
