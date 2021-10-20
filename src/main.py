#!/usr/bin/env python3
import contextlib
import os
import re
import sys

import attrdict
import keras
import keras.callbacks
import keras.models
import keras.optimizers
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import backbones.builder
import callbacks
import data.data_loading
import data.datasets2d
import data.datasets3d
import init
import models.metrabs
import models.util
import parallel_preproc
import tfu
import util
from options import FLAGS, logger
from tfu import TEST, TRAIN, VALID


def train():
    strategy = tf.distribute.MirroredStrategy() if FLAGS.multi_gpu else dummy_strategy()
    n_repl = strategy.num_replicas_in_sync

    #######
    # TRAINING DATA
    #######
    dataset3d = data.datasets3d.get_dataset(FLAGS.dataset)
    joint_info3d = dataset3d.joint_info
    examples3d = get_examples(dataset3d, TRAIN, FLAGS)

    dataset2d = data.datasets2d.get_dataset(FLAGS.dataset2d)
    joint_info2d = dataset2d.joint_info
    examples2d = [*dataset2d.examples[TRAIN], *dataset2d.examples[VALID]]

    if 'many' in FLAGS.dataset:
        if 'aist' in FLAGS.dataset:
            dataset_section_names = 'h36m muco-3dhp surreal panoptic aist_ sailvos'.split()
            roundrobin_sizes = [8, 8, 8, 8, 8, 8]
            roundrobin_sizes = [x * 2 for x in roundrobin_sizes]
        else:
            dataset_section_names = 'h36m muco-3dhp panoptic surreal sailvos'.split()
            roundrobin_sizes = [9, 9, 9, 9, 9]
        example_sections = build_dataset_sections(examples3d, dataset_section_names)
    else:
        example_sections = [examples3d]
        roundrobin_sizes = [FLAGS.batch_size]

    n_completed_steps = get_n_completed_steps(FLAGS.checkpoint_dir, FLAGS.load_path)

    rng = np.random.RandomState(FLAGS.seed)
    data2d = build_dataflow(
        examples2d, data.data_loading.load_and_transform2d, (joint_info2d, TRAIN),
        TRAIN, batch_size=FLAGS.batch_size_2d * n_repl, n_workers=FLAGS.workers,
        rng=util.new_rng(rng), n_completed_steps=n_completed_steps,
        n_total_steps=FLAGS.training_steps)

    data3d = build_dataflow(
        example_sections, data.data_loading.load_and_transform3d, (joint_info3d, TRAIN),
        tfu.TRAIN, batch_size=sum(roundrobin_sizes)//2 * n_repl,
        n_workers=FLAGS.workers,
        rng=util.new_rng(rng), n_completed_steps=n_completed_steps,
        n_total_steps=FLAGS.training_steps, roundrobin_sizes=roundrobin_sizes)

    data_train = tf.data.Dataset.zip((data3d, data2d))
    data_train = data_train.map(lambda batch3d, batch2d: {**batch3d, **batch2d})
    if not FLAGS.multi_gpu:
        data_train = data_train.apply(tf.data.experimental.prefetch_to_device('GPU:0', 2))

    opt = tf.data.Options()
    opt.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    data_train = data_train.with_options(opt)

    #######
    # VALIDATION DATA
    #######
    examples3d_val = get_examples(dataset3d, VALID, FLAGS)

    if FLAGS.validate_period:
        data_val = build_dataflow(
            examples3d_val, data.data_loading.load_and_transform3d,
            (joint_info3d, VALID), VALID, batch_size=FLAGS.batch_size_test * n_repl,
            n_workers=FLAGS.workers, rng=util.new_rng(rng))
        data_val = data_val.with_options(opt)
        validation_steps = int(np.ceil(len(examples3d_val) / (FLAGS.batch_size_test * n_repl)))
    else:
        data_val = None
        validation_steps = None

    #######
    # MODEL
    #######
    with strategy.scope():
        global_step = tf.Variable(n_completed_steps, dtype=tf.int32, trainable=False)
        backbone = backbones.builder.build_backbone()

        model_class = getattr(models, FLAGS.model_class)
        trainer_class = getattr(models, FLAGS.model_class + 'Trainer')

        bone_lengths = (
            dataset3d.trainval_bones if FLAGS.train_on == 'trainval' else dataset3d.train_bones)
        extra_args = [bone_lengths] if FLAGS.model_class.startswith('Model25D') else []
        model = model_class(backbone, joint_info3d, *extra_args)
        trainer = trainer_class(model, joint_info3d, joint_info2d, global_step)
        trainer.compile(optimizer=build_optimizer(global_step, n_repl))
        model.optimizer = trainer.optimizer

    #######
    # CHECKPOINTING
    #######
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, directory=FLAGS.checkpoint_dir, max_to_keep=2, step_counter=global_step,
        checkpoint_interval=FLAGS.checkpoint_period)
    restore_if_ckpt_available(ckpt, ckpt_manager, global_step, FLAGS.init_path)
    trainer.optimizer.iterations.assign(n_completed_steps)

    #######
    # CALLBACKS
    #######
    cbacks = [
        keras.callbacks.LambdaCallback(
            on_train_begin=lambda logs: trainer._train_counter.assign(n_completed_steps),
            on_train_batch_end=lambda batch, logs: ckpt_manager.save(global_step)),
        callbacks.ProgbarCallback(n_completed_steps, FLAGS.training_steps),
        callbacks.WandbCallback(global_step),
        callbacks.TensorBoardCallback(global_step)
    ]

    if FLAGS.finetune_in_inference_mode:
        switch_step = FLAGS.training_steps - FLAGS.finetune_in_inference_mode
        c = callbacks.SwitchToInferenceModeCallback(global_step, switch_step)
        cbacks.append(c)

    #######
    # FITTING
    #######
    try:
        trainer.fit(
            data_train, steps_per_epoch=1, initial_epoch=n_completed_steps,
            epochs=FLAGS.training_steps, verbose=1 if sys.stdout.isatty() else 0,
            callbacks=cbacks, validation_data=data_val, validation_freq=FLAGS.validate_period,
            validation_steps=validation_steps)
        model.save(
            f'{FLAGS.checkpoint_dir}/model', include_optimizer=False, overwrite=True,
            options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))
    except KeyboardInterrupt:
        logger.info('Training interrupted.')
    except tf.errors.ResourceExhaustedError:
        logger.info('Resource Exhausted!')
    finally:
        ckpt_manager.save(global_step, check_interval=False)
        logger.info('Saved checkpoint.')


def build_optimizer(global_step, n_replicas):
    def weight_decay():
        lr_ratio = lr_schedule(global_step) / FLAGS.base_learning_rate
        # Decay the weight decay itself over time the same way as the learning rate is decayed.
        # Division by sqrt(num_training_steps) is taken from the original AdamW paper.
        return FLAGS.weight_decay * lr_ratio / np.sqrt(FLAGS.training_steps)

    def lr():
        return lr_schedule(global_step) / tf.sqrt(tf.cast(n_replicas, tf.float32))

    optimizer = tfa.optimizers.AdamW(weight_decay=weight_decay, learning_rate=lr, epsilon=1e-8)
    # Make sure these exist so checkpoints work properly
    optimizer.iterations
    optimizer.beta_1

    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
        optimizer, dynamic=False, initial_scale=128)
    return optimizer


def build_dataset_sections(examples, section_names):
    sections = {name: [] for name in section_names}
    for ex in examples:
        for name in section_names:
            if name in ex.image_path.lower():
                sections[name].append(ex)
                break
        else:
            raise RuntimeError
    return [sections[name] for name in section_names]


@tf.function
def lr_schedule(global_step):
    training_steps = FLAGS.training_steps
    n_phase1_steps = 0.92 * training_steps
    n_phase2_steps = training_steps - n_phase1_steps
    global_step_float = tf.cast(global_step, tf.float32)
    b = tf.constant(FLAGS.base_learning_rate, tf.float32)

    if global_step_float < n_phase1_steps:
        return tf.keras.optimizers.schedules.ExponentialDecay(
            b, decay_rate=1 / 3, decay_steps=n_phase1_steps, staircase=False)(global_step_float)
    else:
        return tf.keras.optimizers.schedules.ExponentialDecay(
            b * tf.cast(1 / 30, tf.float32), decay_rate=0.3, decay_steps=n_phase2_steps,
            staircase=False)(global_step_float - n_phase1_steps)


def dummy_strategy():
    @contextlib.contextmanager
    def dummy_scope():
        yield

    return attrdict.AttrDict(scope=dummy_scope, num_replicas_in_sync=1)


def export():
    dataset3d = data.datasets3d.get_dataset(FLAGS.dataset)
    ji = dataset3d.joint_info
    del dataset3d
    backbone = backbones.builder.build_backbone()
    model = models.metrabs.Metrabs(backbone, ji)

    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.checkpoint_dir, None)
    restore_if_ckpt_available(ckpt, ckpt_manager, expect_partial=True)

    if FLAGS.load_path:
        load_path = util.ensure_absolute_path(FLAGS.load_path, FLAGS.checkpoint_dir)
    else:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        load_path = ckpt.model_checkpoint_path

    checkpoint_dir = os.path.dirname(load_path)
    out_path = util.ensure_absolute_path(FLAGS.export_file, checkpoint_dir)
    model.save(
        out_path, include_optimizer=False, overwrite=True,
        options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))


def predict():
    dataset3d = data.datasets3d.get_dataset(FLAGS.dataset)
    backbone = backbones.builder.build_backbone()
    model_class = getattr(models, FLAGS.model_class)
    trainer_class = getattr(models, FLAGS.model_class + 'Trainer')
    model_joint_info = data.datasets3d.get_joint_info(FLAGS.model_joints)

    if FLAGS.model_class.startswith('Model25D'):
        bone_dataset = data.datasets3d.get_dataset(FLAGS.bone_length_dataset)
        bone_lengths = (
            bone_dataset.trainval_bones if FLAGS.train_on == 'trainval'
            else bone_dataset.train_bones)
        extra_args = [bone_lengths]
    else:
        extra_args = []
    model = model_class(backbone, model_joint_info, *extra_args)
    trainer = trainer_class(model, model_joint_info)
    trainer.predict_tensor_names = [
        'coords3d_rel_pred', 'coords3d_pred_abs', 'rot_to_world', 'cam_loc', 'image_path']

    if FLAGS.viz:
        trainer.predict_tensor_names += ['image', 'coords3d_true']

    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.checkpoint_dir, None)
    restore_if_ckpt_available(ckpt, ckpt_manager, expect_partial=True)

    examples3d_test = get_examples(dataset3d, tfu.TEST, FLAGS)
    data_test = build_dataflow(
        examples3d_test, data.data_loading.load_and_transform3d,
        (dataset3d.joint_info, TEST), TEST, batch_size=FLAGS.batch_size_test,
        n_workers=FLAGS.workers)
    n_predict_steps = int(np.ceil(len(examples3d_test) / FLAGS.batch_size_test))

    r = trainer.predict(
        data_test, verbose=1 if sys.stdout.isatty() else 0, steps=n_predict_steps)
    r = attrdict.AttrDict(r)
    util.ensure_path_exists(FLAGS.pred_path)

    logger.info(f'Saving predictions to {FLAGS.pred_path}')
    try:
        coords3d_pred = r.coords3d_pred_abs
    except AttributeError:
        coords3d_pred = r.coords3d_rel_pred

    coords3d_pred_world = tf.einsum(
        'nCc, njc->njC', r.rot_to_world, coords3d_pred) + tf.expand_dims(r.cam_loc, 1)
    coords3d_pred_world = models.util.select_skeleton(
        coords3d_pred_world, model_joint_info, FLAGS.output_joints).numpy()
    np.savez(FLAGS.pred_path, image_path=r.image_path, coords3d_pred_world=coords3d_pred_world)


def build_dataflow(
        examples, load_fn, extra_args, learning_phase, batch_size, n_workers, rng=None,
        n_completed_steps=0, n_total_steps=None, n_test_epochs=1, roundrobin_sizes=None):
    if learning_phase == tfu.TRAIN:
        n_total_items = int(n_total_steps * batch_size if n_total_steps is not None else None)
    elif learning_phase == tfu.VALID:
        n_total_items = None
    else:
        n_total_items = int(len(examples) * n_test_epochs)

    dataset = parallel_preproc.parallel_map_as_tf_dataset(
        load_fn, examples, shuffle_before_each_epoch=(learning_phase == tfu.TRAIN),
        extra_args=extra_args, n_workers=n_workers, rng=rng, max_unconsumed=batch_size * 2,
        n_completed_items=n_completed_steps * batch_size, n_total_items=n_total_items,
        roundrobin_sizes=roundrobin_sizes)
    return dataset.batch(batch_size, drop_remainder=(learning_phase == tfu.TRAIN))


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


def get_n_completed_steps(logdir, load_path):
    if load_path is not None:
        return int(re.search('ckpt-(?P<num>\d+)', os.path.basename(load_path))['num'])

    if os.path.exists(f'{logdir}/checkpoint'):
        text = util.read_file(f'{logdir}/checkpoint')
        return int(re.search('model_checkpoint_path: "ckpt-(?P<num>\d+)"', text)['num'])
    else:
        return 0


def restore_if_ckpt_available(
        ckpt, ckpt_manager, global_step_var=None, initial_checkpoint_path=None,
        expect_partial=False):
    resuming_checkpoint_path = FLAGS.load_path
    if resuming_checkpoint_path:
        if resuming_checkpoint_path.endswith('.index'):
            resuming_checkpoint_path = os.path.splitext(resuming_checkpoint_path)[0]
        if not os.path.isabs(resuming_checkpoint_path):
            resuming_checkpoint_path = os.path.join(FLAGS.checkpoint_dir, resuming_checkpoint_path)
    else:
        resuming_checkpoint_path = ckpt_manager.latest_checkpoint

    load_path = resuming_checkpoint_path if resuming_checkpoint_path else initial_checkpoint_path
    if load_path:
        s = ckpt.restore(load_path)
        if expect_partial:
            s.expect_partial()

    if initial_checkpoint_path and not resuming_checkpoint_path:
        global_step_var.assign(0)


def main():
    init.initialize()

    if FLAGS.train:
        train()
    elif FLAGS.predict:
        predict()
    elif FLAGS.export_file:
        export()


if __name__ == '__main__':
    main()
