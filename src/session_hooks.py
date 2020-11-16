import contextlib
import glob
import logging
import math
import multiprocessing as mp
import queue
import re
import signal
import threading
import time

import numpy as np
import tensorflow as tf

import eta
import init
import tfasync
import tfu
import util

mp_spawn = mp.get_context('spawn')


@tfasync.coroutine_hook
async def logger_hook(msg, tensors=()):
    async for values in tfasync.run_iter(tensors):
        formattable_values = [
            util.formattable_array(v) if isinstance(v, np.ndarray) else v
            for v in values]
        logging.info(msg.format(*formattable_values))


def concatenate_atleast_1d(arrays):
    if len(arrays) == 0:
        return np.atleast_1d(arrays)
    elif len(arrays) == 1:
        return np.atleast_1d(*arrays)
    else:
        return np.concatenate(np.atleast_1d(*arrays), axis=0)


@tfasync.coroutine_hook
async def collect_hook(fetches_to_collect=None):
    """Hook that collects the values of specified tensors in every iteration."""
    if fetches_to_collect is None:
        return None

    results = [vals async for vals in tfasync.run_iter(fetches_to_collect)]
    if not isinstance(fetches_to_collect, (list, tuple)):
        return concatenate_atleast_1d(results)

    return [concatenate_atleast_1d(result_column) for result_column in zip(*results)]


@tfasync.coroutine_hook
async def eta_hook(
        n_total_steps, step_tensor=None, init_phase_seconds=60, summary_output_dir=None):
    summary_writer = (tf.compat.v1.summary.FileWriterCache.get(summary_output_dir)
                      if summary_output_dir else None)

    call_times = []  # list of times when hook was called
    remaining_step_counts = []  # list of amount of work remaining at each call

    # (The initial phase can be slower and therefore misleading the ETA calculation
    # so we drop the first `init_phase_seconds` sec of measurements.)
    removed_init = False  # have we removed this initial part already
    global_step_tensor = tf.compat.v1.train.get_or_create_global_step()
    if step_tensor is None:
        step_tensor = global_step_tensor

    async for i_step, i_global_step in tfasync.run_iter([step_tensor, global_step_tensor]):
        call_times.append(time.time())
        remaining_step_counts.append(n_total_steps - i_step)

        if not removed_init and call_times[-1] > call_times[0] + init_phase_seconds:
            del call_times[:-1]
            del remaining_step_counts[:-1]
            removed_init = True

        # We need a few samples to have a good guess.
        if len(call_times) >= 3:
            eta_seconds = eta.eta(call_times, remaining_step_counts)
            eta_str = eta.format_timedelta(eta_seconds)
            progress = i_step / n_total_steps
            logging.info(f'{progress:.0%} complete; {eta_str} remaining.')
            if summary_writer:
                summary_writer.add_summary(tf.compat.v1.Summary(
                    value=[tf.compat.v1.Summary.Value(tag='ETA_hours', simple_value=eta_seconds / 3600)]),
                    global_step=i_global_step)

    if summary_writer:
        summary_writer.flush()


@tfasync.coroutine_hook
async def stop_after_steps_or_seconds_hook(seconds_limit=None, steps_limit=None):
    steps_limit = steps_limit or np.inf
    seconds_limit = seconds_limit or np.inf
    end_time = time.time() + seconds_limit
    step = 0
    while True:
        if time.time() >= end_time or step >= steps_limit:
            break
        step += 1
        await tfasync.run([])

    raise tfasync.RequestLoopStop


@tfasync.coroutine_hook
async def counter_hook(counter):
    await tfasync.run(counter.reset_op)
    while True:
        await tfasync.run(counter.increment_op)


@tfasync.coroutine_hook
async def log_increment_per_sec(name, value_tensor, summary_output_dir=None):
    summary_writer = (tf.compat.v1.summary.FileWriterCache.get(summary_output_dir)
                      if summary_output_dir else None)
    prev_value = None
    prev_time = None
    global_step = tf.compat.v1.train.get_global_step()
    if global_step is None:
        global_step = tf.convert_to_tensor(0)

    async for current_value, step in tfasync.run_iter([value_tensor, global_step]):
        current_time = time.time()
        if prev_value is not None:
            speed = (current_value - prev_value) / (current_time - prev_time)
            logging.info(f'{name}/sec: {speed:.1f}')
            if summary_writer:
                summary_writer.add_summary(tf.compat.v1.Summary(
                    value=[tf.compat.v1.Summary.Value(tag=f'{name}_per_sec', simple_value=speed)]),
                    global_step=step)
                summary_writer.flush()

        prev_value = current_value
        prev_time = current_time


class EvaluationMetric:
    def __init__(self, tensor, name, format_spec, aggregator=None, is_higher_better=True):
        self.tensor = tensor
        self.name = name
        self.format_spec = format_spec
        self.custom_aggregator = aggregator
        self.is_higher_better = is_higher_better

    def get_aggregated_value(self, value):
        if self.custom_aggregator:
            return self.custom_aggregator(value)

        return np.nanmean(value, axis=0)

    def format(self, aggregated_value):
        formatted_val = format(util.formattable_array(aggregated_value), self.format_spec)
        return f'{self.name}: {formatted_val}'

    def is_first_better(self, a, b):
        if self.is_higher_better:
            return a > b
        else:
            return a < b


@tfasync.coroutine_hook
async def validation_hook(
        metrics, checkpoint_path_prefix=None, summary_output_dir=None, max_steps=None,
        max_seconds=None):
    """Evaluates the model and logs the resulting average evaluation metrics.

    Furthermore, if `checkpoint_path_prefix` is given, it also saves a checkpoint whenever there
    is record-low loss. In this case the loss tensor is assumed to be the first metric.
    To reiterate, if checkpointing is desired, the first metric must be a loss, for which lower
    values are better.

    This hook runs a separate validation loop within the existing session.

    Args:
        metric_tensors: The tensors representing the evaluation metrics on the validation set.
        metric_names: The names of the metrics in the same order.
        metric_format_specs: The format specifiers for str.format, for pretty printing in the log.
        checkpoint_path_prefix: Prefix of the path where the best model's checkpoint should be
            saved. If None, no checkpointing is done.
        summary_output_dir: TensorBoard summary directory where the metrics should be written.
        max_steps: The maximum number of steps to run in the validation loop.
        max_seconds: The maximum time to spend on validation. After this the loop is stopped and the
            average metrics are calculated based on the steps that were run.

    Returns:
        A hook that can be used in a training loop.
    """

    saver = tf.compat.v1.train.Saver(max_to_keep=1, save_relative_paths=True)
    summary_writer = (tf.compat.v1.summary.FileWriterCache.get(summary_output_dir)
                      if summary_output_dir else None)

    # Read the best loss from filesystem. The loss value is encoded into the filename of the
    # checkpoint, such as "something-val-0.003348-158585.index"
    if checkpoint_path_prefix:
        paths = glob.glob(checkpoint_path_prefix + '*')
        filename_matches = [re.match(r'.+?-val-(.+?)-.+?\.index', path) for path in paths]
        losses = [float(match[1]) for match in filename_matches if match]
        best_main_metric_value = min(losses) if losses else np.inf
    else:
        best_main_metric_value = np.inf

    # We create some hooks for the internal, nested validation loop. (Hooks within a hook!)
    # First a counter to count steps.
    counter = tfu.get_or_create_counter('validation')
    counter_h = counter_hook(counter)
    eta_h = eta_hook(max_steps, step_tensor=counter.var, init_phase_seconds=30, every_n_secs=10)
    collect_h = collect_hook([m.tensor for m in metrics])
    stop_h = stop_after_steps_or_seconds_hook(seconds_limit=max_seconds, steps_limit=max_steps)
    sigint_h = stop_on_signal_hook(sig=signal.SIGINT)
    sigterm_h = stop_on_signal_hook(sig=signal.SIGTERM, is_additional=True)
    inner_hooks = [counter_h, eta_h, collect_h, stop_h, sigint_h, sigterm_h]
    global_step_tensor = tf.compat.v1.train.get_global_step()

    async for run_context, run_values in tfasync.run_detailed_iter(global_step_tensor):
        global_step_value = run_values.results

        # don't validate after the very first step, not much to validate yet
        if global_step_value == 0:
            continue

        logging.info('Running validation')

        # Run the evaluation loop in the existing session that this validation hook operates in
        tfasync.main_loop(sess=run_context.session, hooks=inner_hooks)

        aggregated_values = [
            metric.get_aggregated_value(result)
            for metric, result in zip(metrics, collect_h.result)]

        # Write to log
        for metric, aggregated_value in zip(metrics, aggregated_values):
            logging.info(metric.format(aggregated_value))

        # Write summaries for TensorBoard
        if summary_writer:
            summary = tfu.scalar_dict_to_summary({
                f'validation/{metric.name}': np.nanmean(value)
                for metric, value in zip(metrics, aggregated_values)})
            summary_writer.add_summary(summary, global_step=global_step_value)
            summary_writer.flush()

        # Save checkpoint if the loss improved
        aggregated_main_value = np.nanmean(aggregated_values[0])
        if metrics[0].is_first_better(aggregated_main_value, best_main_metric_value):
            best_main_metric_value = aggregated_main_value
            logging.info(f'Main metric: {metrics[0].format(aggregated_main_value):} (new record!)')
            if checkpoint_path_prefix:
                saver.save(
                    run_context.session,
                    f'{checkpoint_path_prefix}-val-{aggregated_main_value:.6f}',
                    global_step=global_step_value)
        else:
            logging.info(f'Main metric: {aggregated_main_value:.6f}')


@tfasync.coroutine_hook
async def send_to_worker_hook(
        fetches, worker_main, worker_args=None, worker_kwargs=None, queue_size=30,
        use_threading=True, block=False):
    """Sends the values of `tensors` after each run to a worker process.

    A mp.Queue is used for sending the values.
    In the beginning a new process is created as `worker_proc_main(*worker_args, q=q)`,
    where q is the mp.Queue. Then after each session run, it puts the values
    of `fetches` into the queue.

    If the queue is full, the fetched values are discarded.
    """
    worker_args = worker_args or ()
    worker_kwargs = worker_kwargs or {}
    if use_threading:
        q = queue.Queue(queue_size)
        worker = threading.Thread(
            target=worker_main, args=worker_args, kwargs={'q': q, **worker_kwargs}, daemon=True)
    else:
        # Spawn is used for more memory efficiency
        q = mp_spawn.Queue(queue_size)
        worker = mp_spawn.Process(
            target=util.safe_subprocess_main_with_flags,
            args=(init.FLAGS, worker_main, *worker_args),
            kwargs={'q': q, **worker_kwargs})

    worker.start()
    async for values in tfasync.run_iter(fetches):
        try:
            q.put(values, block=block)
        except queue.Full:
            # logger.debug('Queue Full')
            pass  # discard the fetched values, we don't want to block the session loop.
    q.put(None)
    worker.join()


@tfasync.coroutine_hook
async def rate_limit_hook(max_runs_per_second):
    """Rate limits the loop if it would run faster than `max_runs_per_second`, by blocking the
    loop thread for the necessary amount of time."""

    period = 1 / max_runs_per_second
    last_time = -math.inf
    while True:
        current_time = time.time()
        time_remaining = last_time + period - current_time
        if time_remaining > 0:
            time.sleep(time_remaining)  # blocking sleep, blocks session loop
        last_time = current_time

        await tfasync.run([])


def make_chain_handler(first_handler, second_handler):
    def new_handler(*args, **kwargs):
        try:
            first_handler(*args, **kwargs)
        finally:
            second_handler(*args, **kwargs)

    return new_handler


@contextlib.contextmanager
def use_signal_handler(sig, handler, is_additional=False):
    original_handler = signal.getsignal(sig)
    if is_additional:
        handler = make_chain_handler(handler, original_handler)

    signal.signal(sig, handler)
    yield
    signal.signal(sig, original_handler)


@tfasync.coroutine_hook
async def stop_on_signal_hook(sig=signal.SIGINT, is_additional=False):
    should_stop = False

    # The signal handler simply sets a boolean flag.
    def handle(*args, **kwargs):
        nonlocal should_stop
        should_stop = True

    with use_signal_handler(sig, handle, is_additional):
        async for run_context, run_values in tfasync.run_detailed_iter(()):
            if should_stop:
                logging.warning('-- Interrupted by SIGINT --')
                run_context.request_stop()
                # TODO: Ugly hack, must be refactored
                import parallel_preproc
                parallel_preproc._must_stop = True
