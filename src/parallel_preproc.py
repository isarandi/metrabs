import itertools
import logging
import queue
import threading

import numpy as np
import tensorflow as tf

import tfu
import util


def parallel_map_as_tf_dataset(
        fun, iterable, *, output_types=None, output_shapes=None, shuffle_before_each_epoch=False,
        extra_args=None, n_workers=10, rng=None, max_unconsumed=256, n_completed_items=0,
        n_total_items=None):
    """Maps `fun` to each element of `iterable` and wraps the resulting sequence as
    as a TF Dataset. Elements are processed by parallel workers using mp.

    Args:
        fun: A function that takes an element from seq plus `extra_args` and returns a sequence of
        numpy arrays.
        seq: An iterable holding the inputs.
        output_types: A list of types, describing each output numpy array from `fun`.
            If None, then it is automatically determined by calling `fun` on the first element.
        output_shapes: A list of array shapes, describing each output numpy array from `fun`.
            If None, then it is automatically determined by calling `fun` on the first element.
        shuffle_before_each_epoch: Shuffle the input elements before each epoch. Converts
            `iterable` to a list internally.
        extra_args: extra arguments in addition to an element from `seq`,
            given to `fun` at each call
        n_workers: Number of worker processes for parallelity.
        n_epochs: Number of times to iterate over the `iterable`.

    Returns:
        tf.data.Dataset based on the arrays returned by `fun`.
    """

    extra_args = extra_args or []

    # Automatically determine the output tensor types and shapes by calling the function on
    # the first element
    iterable = list(iterable)
    first_elem = iterable[0]
    if output_types is None or output_shapes is None:
        sample_output = fun(first_elem, *extra_args, rng=np.random.RandomState(0))
        output_shapes, output_types = tfu.get_shapes_and_tf_dtypes(sample_output)

    items = util.iterate_repeatedly(iterable, shuffle_before_each_epoch, util.new_rng(rng))

    # If we are restoring from a checkpoint and have already completed some
    # training steps for that checkpoint, then we need to advance the RNG
    # accordingly, to continue exactly where we left off.
    iter_rng = util.new_rng(rng)
    util.advance_rng(iter_rng, n_completed_items)
    logging.debug(f'n_total_items: {n_total_items}, n_completed_items: {n_completed_items}')
    items = itertools.islice(items, n_completed_items, n_total_items)

    if n_workers == 1:
        def gen():
            for item in items:
                logging.debug('yielding')
                yield fun(item, *extra_args, util.new_rng(iter_rng))
            logging.debug('ended')
    else:
        pool = tfu.get_pool(n_workers)
        gen = parallel_map_as_generator(
            fun, items, extra_args, pool, rng=iter_rng, max_unconsumed=max_unconsumed)

    return tf.data.Dataset.from_generator(gen, output_types, output_shapes)

_must_stop = False

def parallel_map_as_generator(
        fun, items, extra_args, pool, max_unconsumed=256, rng=None):
    semaphore = threading.Semaphore(max_unconsumed)
    q = queue.Queue()
    end_of_sequence_marker = object()

    def producer():
        for i_item, item in enumerate(items):
            semaphore.acquire()
            if _must_stop:
                return
            q.put(pool.apply_async(fun, (item, *extra_args, util.new_rng(rng))))

        logging.debug('Putting end-of-seq')
        q.put(end_of_sequence_marker)

    def consumer():
        while True:
            future_or_end = q.get()
            if future_or_end is end_of_sequence_marker or _must_stop:
                logging.debug('Received end-of-seq')
                return
            else:
                value = tuple(future_or_end.get())
                semaphore.release()
                yield value

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    return consumer
