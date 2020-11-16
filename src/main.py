#!/usr/bin/env python3

import init

# This `pass` keeps `init` on top, which needs to be imported first.
pass
import shutil
import paths
import logging
import os
import os.path
import re

import numpy as np
from attrdict import AttrDict

import data.datasets3d
import data.datasets2d
import data.data_loading
import helpers

import model.bone_length_based_backproj
import model.metrabs
import model.metro
import model.twofive
import session_hooks
import tfasync
import tensorflow as tf
import tfu
import tfu3d
import util
import util3d
from options import FLAGS
from session_hooks import EvaluationMetric
from tfu import TEST, TRAIN, VALID
import attrdict

try:
    import tensorflow_addons
except:
    pass


def main():
    init.initialize()
    if FLAGS.train:
        train()
    if FLAGS.test:
        test()
    if FLAGS.export_file:
        export()


def train():
    logging.info('Training phase.')
    rng = np.random.RandomState(FLAGS.seed)
    n_done_steps = get_number_of_already_completed_steps(FLAGS.logdir, FLAGS.load_path)

    t_train = build_graph(
        TRAIN, rng=util.new_rng(rng), n_epochs=FLAGS.epochs, n_done_steps=n_done_steps)
    logging.info(f'Number of trainable parameters: {tfu.count_trainable_params():,}')
    t_valid = (build_graph(VALID, shuffle=True, rng=util.new_rng(rng))
               if FLAGS.validate_period else None)

    helpers.run_train_loop(
        t_train.train_op, checkpoint_dir=FLAGS.checkpoint_dir, load_path=FLAGS.load_path,
        hooks=make_training_hooks(t_train, t_valid), init_fn=get_init_fn())
    logging.info('Ended training.')


def test():
    logging.info('Test (scripts) phase.')
    tf.compat.v1.reset_default_graph()
    t = build_graph(TEST, n_epochs=1, shuffle=False)

    test_counter = tfu.get_or_create_counter('testc')
    counter_hook = session_hooks.counter_hook(test_counter)
    example_hook = session_hooks.log_increment_per_sec(
        'Examples', test_counter.var * FLAGS.batch_size_test, None, every_n_secs=FLAGS.hook_seconds)

    hooks = [example_hook, counter_hook]
    dataset = data.datasets3d.get_dataset(FLAGS.dataset)
    if FLAGS.gui:
        plot_hook = session_hooks.send_to_worker_hook(
            [tfu.std_to_nhwc(t.x[0]), t.coords3d_pred[0], t.coords3d_true[0]],
            util3d.plot3d_worker,
            worker_args=[dataset.joint_info.stick_figure_edges],
            worker_kwargs=dict(batched=False, interval=100, has_ground_truth=True),
            every_n_steps=1, use_threading=False)
        rate_limit_hook = session_hooks.rate_limit_hook(0.5)
        hooks.append(plot_hook)
        hooks.append(rate_limit_hook)

    fetch_names = [
        'image_path', 'coords3d_true_orig_cam', 'coords3d_pred_orig_cam', 'coords3d_true_world',
        'coords3d_pred_world', 'activity_name', 'scene_name', 'joint_validity_mask']
    fetch_tensors = {fetch_name: t[fetch_name] for fetch_name in fetch_names}

    global_init_op = tf.compat.v1.global_variables_initializer()
    local_init_op = tf.compat.v1.local_variables_initializer()

    def init_fn(_, sess):
        sess.run([global_init_op, local_init_op, test_counter.reset_op])

    f = helpers.run_eval_loop(
        fetches_to_collect=fetch_tensors, load_path=FLAGS.load_path,
        checkpoint_dir=FLAGS.checkpoint_dir, hooks=hooks, init_fn=init_fn)
    save_results(f)


def save_results(f):
    ordered_indices = np.argsort(f.image_path)
    util.ensure_path_exists(FLAGS.pred_path)
    logging.info(f'Saving predictions to {FLAGS.pred_path}')
    np.savez(
        FLAGS.pred_path,
        image_path=f.image_path[ordered_indices],
        coords3d_true=f.coords3d_true_orig_cam[ordered_indices],
        coords3d_pred=f.coords3d_pred_orig_cam[ordered_indices],
        coords3d_true_world=f.coords3d_true_world[ordered_indices],
        coords3d_pred_world=f.coords3d_pred_world[ordered_indices],
        activity_name=f.activity_name[ordered_indices],
        scene_name=f.scene_name[ordered_indices],
        joint_validity_mask=f.joint_validity_mask[ordered_indices],
    )


def build_graph(
        learning_phase, n_epochs=None, shuffle=None, drop_remainder=None, rng=None, n_done_steps=0):
    tfu.set_is_training(learning_phase == TRAIN)

    t = AttrDict(global_step=tf.compat.v1.train.get_or_create_global_step())

    dataset3d = data.datasets3d.get_dataset(FLAGS.dataset)
    examples = helpers.get_examples(dataset3d, learning_phase, FLAGS)
    t.n_examples = len(examples)
    phase_name = 'training' if tfu.is_training() else 'validation'
    logging.info(f'Number of {phase_name} examples: {t.n_examples:,}')

    n_total_steps = None
    if n_epochs is not None:
        batch_size = FLAGS.batch_size if learning_phase == TRAIN else FLAGS.batch_size_test
        n_total_steps = (len(examples) * n_epochs) // batch_size

    if rng is None:
        rng = np.random.RandomState()

    if tfu.is_training() and FLAGS.train_mixed:
        dataset2d = data.datasets2d.get_dataset(FLAGS.dataset2d)
        examples2d = [*dataset2d.examples[tfu.TRAIN], *dataset2d.examples[tfu.VALID]]
        build_mixed_batch(
            t, dataset3d, dataset2d, examples, examples2d, learning_phase,
            batch_size3d=FLAGS.batch_size, batch_size2d=FLAGS.batch_size_2d,
            shuffle=shuffle, rng=rng, max_unconsumed=FLAGS.max_unconsumed,
            n_done_steps=n_done_steps, n_total_steps=n_total_steps)
    else:
        batch_size = FLAGS.batch_size if learning_phase == TRAIN else FLAGS.batch_size_test
        helpers.build_input_batch(
            t, examples, data.data_loading.load_and_transform3d,
            (dataset3d.joint_info, learning_phase), learning_phase, batch_size,
            FLAGS.workers, shuffle=shuffle, drop_remainder=drop_remainder, rng=rng,
            max_unconsumed=FLAGS.max_unconsumed, n_done_steps=n_done_steps,
            n_total_steps=n_total_steps)
        (t.image_path, t.x, t.coords3d_true, t.coords2d_true, t.inv_intrinsics,
         t.rot_to_orig_cam, t.rot_to_world, t.cam_loc, t.joint_validity_mask,
         t.is_joint_in_fov, t.activity_name, t.scene_name) = t.batch

    if FLAGS.scale_recovery == 'metrabs':
        model.metrabs.build_metrabs_model(dataset3d.joint_info, t)
    elif FLAGS.scale_recovery == 'metro':
        model.metro.build_metro_model(dataset3d.joint_info, t)
    else:
        model.twofive.build_25d_model(dataset3d.joint_info, t)

    if 'coords3d_true' in t:
        build_eval_metrics(t)

    if learning_phase == TRAIN:
        build_train_op(t)
        build_summaries(t)
    return t


@tfu.in_name_scope('InputPipeline')
def build_mixed_batch(
        t, dataset3d, dataset2d, examples3d, examples2d, learning_phase,
        batch_size3d=None, batch_size2d=None, shuffle=None, rng=None,
        max_unconsumed=256, n_done_steps=0, n_total_steps=None):
    if shuffle is None:
        shuffle = learning_phase == TRAIN

    if rng is None:
        rng = np.random.RandomState()
    rng_2d = util.new_rng(rng)
    rng_3d = util.new_rng(rng)

    (t.image_path_2d, t.x_2d, t.coords2d_true2d,
     t.joint_validity_mask2d) = helpers.build_input_batch(
        t, examples2d, data.data_loading.load_and_transform2d,
        (dataset2d.joint_info, learning_phase), learning_phase, batch_size2d, FLAGS.workers,
        shuffle=shuffle, rng=rng_2d, max_unconsumed=max_unconsumed,
        n_done_steps=n_done_steps, n_total_steps=n_total_steps)

    (t.image_path, t.x, t.coords3d_true, t.coords2d_true, t.inv_intrinsics,
     t.rot_to_orig_cam, t.rot_to_world, t.cam_loc, t.joint_validity_mask,
     t.is_joint_in_fov, t.activity_name, t.scene_name) = helpers.build_input_batch(
        t, examples3d, data.data_loading.load_and_transform3d,
        (dataset3d.joint_info, learning_phase), learning_phase, batch_size3d, FLAGS.workers,
        shuffle=shuffle, rng=rng_3d, max_unconsumed=max_unconsumed,
        n_done_steps=n_done_steps, n_total_steps=n_total_steps)


@tfu.in_name_scope('Optimizer')
def build_train_op(t):
    t.global_step = tf.compat.v1.train.get_or_create_global_step()
    t.learning_rate = learning_rate_schedule(
        t.global_step, t.n_examples / FLAGS.batch_size, FLAGS.learning_rate, FLAGS)

    n_steps_total = t.n_examples / FLAGS.batch_size * FLAGS.epochs

    # Formula as per AdamW paper [Loshchilov & Hutter ICLR'19, arXiv:1711.05101]
    weight_decay = (FLAGS.weight_decay * (
            t.learning_rate / FLAGS.learning_rate) / n_steps_total ** 0.5)
    try:
        AdamW = tensorflow_addons.optimizers.AdamW
    except:
        AdamW = tf.contrib.opt.AdamWOptimizer

    t.optimizer = AdamW(
        weight_decay=weight_decay, learning_rate=t.learning_rate, epsilon=FLAGS.epsilon)

    trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    logging.info(f'Update op count: {len(update_ops)}')

    def minimize(optimizer, loss, var_list, update_list, step=t.global_step):
        gradients = tfu.gradients_with_loss_scaling(loss, var_list, 256)
        gradients_and_vars = list(zip(gradients, var_list))
        m = optimizer.apply_gradients(gradients_and_vars)
        with tf.control_dependencies([
            tf.compat.v1.train.get_or_create_global_step(), tf.compat.v1.train.get_global_step()]):
            increment_global_step = tf.compat.v1.assign_add(step, 1)
        return tf.group(m, *update_list, increment_global_step)

    t.train_op = minimize(t.optimizer, t.loss, trainable_vars, update_ops)


def learning_rate_schedule(global_step, steps_per_epoch, base_learning_rate, flags):
    all_steps = flags.epochs * steps_per_epoch
    n_phase2_steps = 2 / 25 * all_steps
    n_phase1_steps = all_steps - n_phase2_steps

    global_step_float = tf.cast(global_step, tf.float32)
    phase1_lr = tf.compat.v1.train.exponential_decay(
        tf.cast(base_learning_rate, tf.float32), global_step=global_step_float, decay_rate=1 / 3,
        decay_steps=n_phase1_steps, staircase=False)

    phase2_lr = tf.compat.v1.train.exponential_decay(
        tf.cast(base_learning_rate / 30, tf.float32),
        global_step=global_step_float - n_phase1_steps,
        decay_rate=0.3, decay_steps=n_phase2_steps, staircase=False)

    return tf.where(global_step_float < n_phase1_steps, phase1_lr, phase2_lr)


@tfu.in_name_scope('EvalMetrics')
def build_eval_metrics(t):
    rootrelative_diff = tfu3d.root_relative(t.coords3d_pred - t.coords3d_true)
    dist = tf.norm(rootrelative_diff, axis=-1)
    t.mean_error = tfu.reduce_mean_masked(dist, t.joint_validity_mask)
    t.coords3d_pred_procrustes = tfu3d.rigid_align(
        t.coords3d_pred, t.coords3d_true,
        joint_validity_mask=t.joint_validity_mask, scale_align=True)

    rootrelative_diff_procrust = tfu3d.root_relative(t.coords3d_pred_procrustes - t.coords3d_true)
    dist_procrustes = tf.norm(rootrelative_diff_procrust, axis=-1)
    t.mean_error_procrustes = tfu.reduce_mean_masked(dist_procrustes, t.joint_validity_mask)

    threshold = np.float32(150)
    auc_score = tf.maximum(np.float32(0), 1 - dist / threshold)
    t.auc_for_nms = tfu.reduce_mean_masked(auc_score, t.joint_validity_mask, axis=0)
    t.mean_auc = tfu.reduce_mean_masked(auc_score, t.joint_validity_mask)

    is_correct = tf.cast(dist <= threshold, tf.float32)
    t.pck = tfu.reduce_mean_masked(is_correct, t.joint_validity_mask, axis=0)
    t.mean_pck = tfu.reduce_mean_masked(is_correct, t.joint_validity_mask)


def build_summaries(t):
    t.epoch = t.global_step // (t.n_examples // FLAGS.batch_size)
    opnames = 'epoch learning_rate mean_error mean_error_procrustes loss loss22d loss23d loss32d ' \
              'loss3d loss3d_abs'.split()
    scalar_ops = ({name: t[name] for name in opnames if name in t})
    prefix = 'training' if tfu.is_training() else 'validation'
    summaries = [tf.compat.v1.summary.scalar(f'{prefix}/{name}', op, collections=[])
                 for name, op in scalar_ops.items()]
    t.summary_op = tf.compat.v1.summary.merge(summaries)


def make_training_hooks(t_train, t_valid):
    saver = tf.compat.v1.train.Saver(max_to_keep=2, save_relative_paths=True)

    checkpoint_state = tf.train.get_checkpoint_state(FLAGS.logdir)
    if checkpoint_state:
        saver.recover_last_checkpoints(checkpoint_state.all_model_checkpoint_paths)

    global_step_tensor = tf.compat.v1.train.get_or_create_global_step()
    checkpoint_hook = tf.estimator.CheckpointSaverHook(FLAGS.logdir, saver=saver, save_secs=30 * 60)
    total_batch_size = FLAGS.batch_size * (2 if FLAGS.train_mixed else 1)
    example_counter_hook = session_hooks.log_increment_per_sec(
        'Training images', t_train.global_step * total_batch_size, every_n_secs=FLAGS.hook_seconds,
        summary_output_dir=FLAGS.logdir)

    i_epoch = t_train.global_step // (t_train.n_examples // FLAGS.batch_size)
    logger_hook = session_hooks.logger_hook(
        'Epoch {:03d}, global step {:07d}. Loss: {:.15e}',
        [i_epoch, t_train.global_step, t_train.loss], every_n_steps=1)

    hooks = [example_counter_hook, logger_hook, checkpoint_hook]

    if FLAGS.epochs:
        eta_hook = session_hooks.eta_hook(
            n_total_steps=(t_train.n_examples * FLAGS.epochs) // FLAGS.batch_size,
            every_n_secs=600, summary_output_dir=FLAGS.logdir)
        hooks.append(eta_hook)

    if FLAGS.validate_period:
        every_n_steps = (
            int(np.round(FLAGS.validate_period * (t_train.n_examples // FLAGS.batch_size))))

        max_valid_steps = np.ceil(t_valid.n_examples / FLAGS.batch_size_test)
        summary_output_dir = FLAGS.logdir if FLAGS.tensorboard else None

        metrics = [
            EvaluationMetric(t_valid.mean_error, 'MPJPE', '.3f', is_higher_better=False),
            EvaluationMetric(t_valid.mean_error_procrustes, 'MPJPE-procrustes', '.3f',
                             is_higher_better=False),
            EvaluationMetric(t_valid.mean_pck, '3DPCK@150mm', '.3%'),
            EvaluationMetric(t_valid.mean_auc, 'AUC', '.3%'),
        ]

        validation_hook = session_hooks.validation_hook(
            metrics, summary_output_dir=summary_output_dir, max_steps=max_valid_steps,
            max_seconds=120, every_n_steps=every_n_steps, _step_tensor=global_step_tensor)
        hooks.append(validation_hook)

    if FLAGS.tensorboard:
        other_summary_ops = [a for a in [tf.compat.v1.summary.merge_all()] if a is not None]
        summary_hook = tf.estimator.SummarySaverHook(
            save_steps=1, output_dir=FLAGS.logdir,
            summary_op=tf.compat.v1.summary.merge([*other_summary_ops, t_train.summary_op]))
        summary_hook = tfasync.PeriodicHookWrapper(
            summary_hook, every_n_steps=10, step_tensor=global_step_tensor)
        hooks.append(summary_hook)

    if FLAGS.gui:
        dataset = data.datasets3d.get_dataset(FLAGS.dataset)
        plot_hook = session_hooks.send_to_worker_hook(
            [tfu.std_to_nhwc(t_train.x[0]), t_train.coords3d_pred[0], t_train.coords3d_true[0]],
            util3d.plot3d_worker, worker_args=[dataset.joint_info.stick_figure_edges],
            worker_kwargs=dict(batched=False, interval=100), every_n_secs=FLAGS.hook_seconds,
            use_threading=False)
        hooks.append(plot_hook)
        if 'coords3d_pred2d' in t_train:
            plot_hook = session_hooks.send_to_worker_hook(
                [tfu.std_to_nhwc(t_train.x_2d[0]), t_train.coords3d_pred2d[0],
                 t_train.coords3d_pred2d[0]],
                util3d.plot3d_worker, worker_args=[dataset.joint_info.stick_figure_edges],
                worker_kwargs=dict(batched=False, interval=100, has_ground_truth=False),
                every_n_secs=FLAGS.hook_seconds, use_threading=False)
            hooks.append(plot_hook)

    return hooks


def get_init_fn():
    if FLAGS.init == 'scratch':
        return None
    elif FLAGS.init != 'pretrained':
        raise NotImplementedError

    if FLAGS.architecture.startswith('resnet_v2'):
        default_weight_subpath = f'{FLAGS.architecture}_2017_04_14/{FLAGS.architecture}.ckpt'
    else:
        raise Exception(
            f'No default pretrained weights configured for architecture {FLAGS.architecture}')

    if FLAGS.init_path:
        weight_path = FLAGS.init_path
        checkpoint_scope = f'MainPart/{FLAGS.architecture}'
    else:
        weight_path = f'{paths.DATA_ROOT}/pretrained/{default_weight_subpath}'
        if not os.path.exists(weight_path) and not os.path.exists(weight_path + '.index'):
            download_pretrained_weights()
        checkpoint_scope = FLAGS.architecture

    loaded_scope = f'MainPart/{FLAGS.architecture}'
    do_not_load = ['Adam', 'Momentum', 'noload']
    if FLAGS.init_logits_random:
        do_not_load.append('logits')

    return tfu.make_pretrained_weight_loader(
        weight_path, loaded_scope, checkpoint_scope, do_not_load)


def download_pretrained_weights():
    import urllib.request
    import tarfile

    logging.info(f'Downloading ImageNet pretrained weights for {FLAGS.architecture}')
    filename = f'{FLAGS.architecture}_2017_04_14.tar.gz'
    target_path = f'{paths.DATA_ROOT}/pretrained/{FLAGS.architecture}_2017_04_14/{filename}'
    util.ensure_path_exists(target_path)
    urllib.request.urlretrieve(f'http://download.tensorflow.org/models/{filename}', target_path)
    with tarfile.open(target_path) as f:
        f.extractall(f'{paths.DATA_ROOT}/pretrained/{FLAGS.architecture}_2017_04_14')
    os.remove(target_path)


def get_number_of_already_completed_steps(logdir, load_path):
    """Find out how many training steps have already been completed
     in case we are resuming training from a checkpoint file."""

    if load_path is not None:
        return int(re.search(r'model.ckpt-(?P<num>\d+)(\.|$)', os.path.basename(load_path))['num'])

    if os.path.exists(f'{logdir}/checkpoint'):
        text = util.read_file(f'{logdir}/checkpoint')
        return int(re.search(r'model_checkpoint_path: "model.ckpt-(?P<num>\d+)"', text)['num'])
    else:
        return 0


def export():
    logging.info('Exporting model file.')
    tf.compat.v1.reset_default_graph()

    t = attrdict.AttrDict()
    t.x = tf.compat.v1.placeholder(
        shape=[None, FLAGS.proc_side, FLAGS.proc_side, 3], dtype=tfu.get_dtype())
    t.x = tfu.nhwc_to_std(t.x)

    is_absolute_model = FLAGS.scale_recovery in ('metrabs',)

    if is_absolute_model:
        intrinsics_tensor = tf.compat.v1.placeholder(shape=[None, 3, 3], dtype=tf.float32)
        t.inv_intrinsics = tf.linalg.inv(intrinsics_tensor)
    else:
        intrinsics_tensor = None

    joint_info = data.datasets3d.get_dataset(FLAGS.dataset).joint_info

    if FLAGS.scale_recovery == 'metrabs':
        model.metrabs.build_metrabs_inference_model(joint_info, t)
    elif FLAGS.scale_recovery == 'metro':
        model.metro.build_metro_inference_model(joint_info, t)
    else:
        model.twofive.build_25d_inference_model(joint_info, t)

    # Convert to the original joint order as defined in the original datasets
    # (i.e. put the pelvis back to its place from the last position,
    # because this codebase normally uses the last position for the pelvis in all cases for
    # consistency)
    if FLAGS.dataset == 'many':
        selected_joint_ids = [23, *range(23)] if FLAGS.export_smpl else [*range(73)]
    elif FLAGS.dataset == 'h36m':
        selected_joint_ids = [16, *range(16)]
    else:
        assert FLAGS.dataset in ('mpi_inf_3dhp', 'mupots') or 'muco' in FLAGS.dataset
        selected_joint_ids = [*range(14), 17, 14, 15]

    t.coords3d_pred = tf.gather(t.coords3d_pred, selected_joint_ids, axis=1)
    joint_info = joint_info.select_joints(selected_joint_ids)

    if FLAGS.load_path:
        load_path = util.ensure_absolute_path(FLAGS.load_path, FLAGS.checkpoint_dir)
    else:
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        load_path = checkpoint.model_checkpoint_path
    checkpoint_dir = os.path.dirname(load_path)
    out_path = util.ensure_absolute_path(FLAGS.export_file, checkpoint_dir)

    sm = tf.compat.v1.saved_model
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, load_path)
        inputs = (dict(image=t.x, intrinsics=intrinsics_tensor) if is_absolute_model
                  else dict(image=t.x))

        signature_def = sm.signature_def_utils.predict_signature_def(
            inputs=inputs, outputs=dict(poses=t.coords3d_pred))
        os.mkdir(out_path)
        builder = sm.builder.SavedModelBuilder(out_path)
        builder.add_meta_graph_and_variables(
            sess, ['serve'], signature_def_map=dict(serving_default=signature_def))
        builder.save()

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.enable_eager_execution()
    crop_model = tf.saved_model.load(out_path)
    shutil.rmtree(out_path)

    wrapper_class = (ExportedAbsoluteModel if is_absolute_model else ExportedRootRelativeModel)
    wrapped_model = wrapper_class(crop_model, joint_info)
    tf.saved_model.save(wrapped_model, out_path)


class ExportedAbsoluteModel(tf.Module):
    def __init__(self, crop_model, joint_info):
        super().__init__()
        self.crop_model = crop_model
        self.predict_crop = self.crop_model.signatures['serving_default']
        self.joint_names = tf.Variable(np.array(joint_info.names), trainable=False)
        self.joint_edges = tf.Variable(np.array(joint_info.stick_figure_edges), trainable=False)

        @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, None, None, 3), dtype=tfu.get_dtype()),
            tf.TensorSpec(shape=(None, 3, 3), dtype=tf.float32)])
        def __call__(image, intrinsic_matrix):
            return self.predict_crop(image=image, intrinsics=intrinsic_matrix)['poses']

        self.__call__ = __call__


class ExportedRootRelativeModel(tf.Module):
    def __init__(self, crop_model, joint_info):
        super().__init__()
        self.crop_model = crop_model
        self.predict_crop = self.crop_model.signatures['serving_default']
        self.joint_names = tf.Variable(np.array(joint_info.names), trainable=False)
        self.joint_edges = tf.Variable(np.array(joint_info.stick_figure_edges), trainable=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.uint8)])
    def __call__(self, image):
        return self.predict_crop(image=image)['poses']


if __name__ == '__main__':
    main()
