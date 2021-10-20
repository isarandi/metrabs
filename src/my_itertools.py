import numpy as np

import util


def roundrobin(iterables, sizes):
    iterators = [iter(iterable) for iterable in iterables]
    while True:
        for iterator, size in zip(iterators, sizes):
            for i in range(size):
                try:
                    yield next(iterator)
                except StopIteration:
                    return


def iterate_repeatedly(seq, shuffle_before_each_epoch=False, rng=None):
    """Iterates over and yields the elements of `iterable` over and over.
    If `shuffle_before_each_epoch` is True, the elements are put in a list and shuffled before
    every pass over the data, including the first."""

    if rng is None:
        rng = np.random.RandomState()

    # create a (shallow) copy so shuffling only applies to the copy.
    seq = list(seq)
    rng.shuffle(seq)
    yield from seq

    while True:
        if shuffle_before_each_epoch:
            rng.shuffle(seq)
        yield from seq


def roundrobin_iterate_repeatedly(
        seqs, roundrobin_sizes, shuffle_before_each_epoch=False, rng=None):
    iters = [iterate_repeatedly(seq, shuffle_before_each_epoch, util.new_rng(rng)) for seq in seqs]
    return roundrobin(iters, roundrobin_sizes)
