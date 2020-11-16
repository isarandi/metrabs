import logging
import os
import os.path

import attrdict
import tensorflow as tf

import parallel_preproc
import session_hooks
import tfasync
import tfu
import util


def train(
        graph_fn, hook_fn=None, checkpoint_dir=None, load_path=None, init_fn=None,
        validation=False):
    logging.info('Training phase.')
    t_train = graph_fn(tfu.TRAIN, reuse=tf.compat.v1.AUTO_REUSE)
    logging.info(f'Number of trainable parameters: {tfu.count_trainable_params():,}')
    # logging.info(f'Trainable parameters: {tfu.describe_trainable_params()}')

    t_valid = (graph_fn(tfu.VALID, reuse=tf.compat.v1.AUTO_REUSE) if validation else None)
    hooks = hook_fn(t_train, t_valid)

    run_train_loop(
        t_train.train_op, checkpoint_dir=checkpoint_dir, load_path=load_path,
        hooks=hooks, init_fn=init_fn)


@tfu.in_name_scope('InputPipeline')
def build_input_batch(
        t, examples, load_fn, extra_args, learning_phase, batch_size, n_workers, shuffle=None,
        drop_remainder=None, rng=None, max_unconsumed=256, n_done_steps=0, n_total_steps=None,
        n_test_epochs=1):
    if shuffle is None:
        shuffle = learning_phase in (tfu.TRAIN, tfu.VALID)

    if learning_phase == tfu.TRAIN:
        n_total_items = int(n_total_steps * batch_size if n_total_steps is not None else None)
    elif learning_phase == tfu.VALID:
        n_total_items = None
    else:
        n_total_items = int(len(examples) * n_test_epochs)

    dataset = parallel_preproc.parallel_map_as_tf_dataset(
        load_fn, examples, shuffle_before_each_epoch=shuffle, extra_args=extra_args,
        n_workers=n_workers, rng=rng, max_unconsumed=max_unconsumed,
        n_completed_items=n_done_steps * batch_size, n_total_items=n_total_items)

    if drop_remainder is None:
        drop_remainder = (learning_phase == tfu.TRAIN)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(2)

    # Prefetching to the GPU makes the training faster and does not seem to reduce
    # reproducibility.
    if os.environ.get('CUDA_VISIBLE_DEVICES', 'something') != '' and learning_phase == tfu.TRAIN:
        dataset = dataset.apply(tf.data.experimental.prefetch_to_device('device:XLA_GPU:0', 3))

    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    t.batch = iterator.get_next()
    return t.batch


def get_examples(dataset, learning_phase, flags):
    if learning_phase == tfu.TRAIN:
        str_example_phase = flags.train_on
    elif learning_phase == tfu.VALID:
        str_example_phase = flags.validate_on
    elif learning_phase == tfu.TEST:
        str_example_phase = flags.test_on
    else:
        raise Exception(f'No such learning_phase as {learning_phase}')

    if str_example_phase == 'train':
        examples = dataset.examples[tfu.TRAIN]
    elif str_example_phase == 'valid':
        examples = dataset.examples[tfu.VALID]
    elif str_example_phase == 'test':
        examples = dataset.examples[tfu.TEST]
    elif str_example_phase == 'trainval':
        examples = [*dataset.examples[tfu.TRAIN], *dataset.examples[tfu.VALID]]
    else:
        raise Exception(f'No such phase as {str_example_phase}')
    return examples


def run_train_loop(
        train_op, sess=None, checkpoint_dir=None, load_path=None, max_steps=None, hooks=(),
        init_fn=None):
    sess_creator = None if sess else make_session_creator(checkpoint_dir, load_path, init_fn)
    if max_steps:
        stop_hook = session_hooks.stop_after_steps_or_seconds_hook(steps_limit=max_steps)
        hooks = [stop_hook, *hooks]

    hooks = [*hooks, session_hooks.stop_on_signal_hook()]

    tfasync.main_loop(sess=sess, sess_creator=sess_creator, ops=train_op, hooks=hooks)


def run_eval_loop(
        sess=None, fetches_to_collect=None, other_ops=(), hooks=(), checkpoint_dir=None,
        load_path=None, max_steps=None, max_seconds=None, init_fn=None):
    if isinstance(fetches_to_collect, dict):
        keys, values = zip(*fetches_to_collect.items())
        results = run_eval_loop(
            sess, list(values), other_ops, hooks, checkpoint_dir, load_path, max_steps, max_seconds)
        return attrdict.AttrDict(dict(zip(keys, results)))

    sess_creator = None if sess else make_session_creator(checkpoint_dir, load_path, init_fn)
    collect_hook = session_hooks.collect_hook(fetches_to_collect)
    hooks = [collect_hook, *hooks]
    if max_seconds or max_steps:
        stop_hook = session_hooks.stop_after_steps_or_seconds_hook(max_seconds, max_steps)
        hooks.append(stop_hook)

    tfasync.main_loop(sess=sess, sess_creator=sess_creator, ops=other_ops, hooks=hooks)
    return collect_hook.result


def make_session_creator(checkpoint_dir=None, load_path=None, init_fn=None):
    config = tf.compat.v1.ConfigProto()
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = False

    scaffold = tf.compat.v1.train.Scaffold(init_fn=init_fn)

    if load_path:
        return tf.compat.v1.train.ChiefSessionCreator(
            checkpoint_filename_with_path=load_path, config=config, scaffold=scaffold)
    elif checkpoint_dir:
        return tf.compat.v1.train.ChiefSessionCreator(
            checkpoint_dir=checkpoint_dir, config=config, scaffold=scaffold)
    else:
        return tf.compat.v1.train.ChiefSessionCreator(config=config, scaffold=scaffold)
