import os

import keras.callbacks
import tensorflow as tf
import wandb

import util
from options import FLAGS


class WandbCallback(keras.callbacks.Callback):
    def __init__(self, global_step_var):
        super().__init__()
        self.global_step_var = global_step_var

    def on_train_begin(self, logs=None):
        id_path = f'{FLAGS.logdir}/run_id'
        run_id = util.read_file(id_path) if os.path.exists(id_path) else wandb.util.generate_id()
        util.write_file(run_id, id_path)
        wandb.init(
            name=FLAGS.logdir.split('/')[-1], project='metrabs', config=FLAGS,
            dir=f'{FLAGS.logdir}', id=run_id, resume='allow')

    def on_epoch_end(self, epoch, logs=None):
        step = self.global_step_var.value()
        # Only report training batch metrics for every 30th step.
        if step % 30 != 0:
            logs = {k: v for k, v in logs.items() if k.startswith('val_')}
        if logs:
            wandb.log(logs, step=step, commit=True)


class TensorBoardCallback(keras.callbacks.Callback):
    def __init__(self, global_step_var):
        super().__init__()
        self.global_step_var = global_step_var
        self.writer = tf.summary.create_file_writer(FLAGS.logdir)

    def on_epoch_end(self, epoch, logs=None):
        step = self.global_step_var.value()
        # Only report training batch metrics for every 30th step.
        # if step % 30 != 0:
        #    logs = {k: v for k, v in logs.items() if k.startswith('val_')}
        val_mapping = dict(
            mean_auc='AUC', mean_pck='3DPCK@150mm', mean_error_abs='A-MPJPE',
            pck_wrists='3DPCK-wri@150mm', mean_error='MPJPE',
            auc_wrists='AUC-wri',
            mean_error_procrustes='MPJPE-procrustes', mean_error_2d='2D-MPJPE')
        train_mapping = dict(loss='batch_loss')
        if logs:
            with self.writer.as_default():
                for k, v in logs.items():
                    k = k.replace('val_metrics/', 'validation/')
                    k = k.replace('metrics/', 'training/Summaries/')
                    parts = k.split('/')
                    if k.startswith('training'):
                        parts[-1] = train_mapping.get(parts[-1], parts[-1])
                    else:
                        parts[-1] = val_mapping.get(parts[-1], parts[-1])
                    k = '/'.join(parts)
                    tf.summary.scalar(k, v, step=tf.cast(step, tf.int64))
            self.writer.flush()


class ProgbarCallback(keras.callbacks.ProgbarLogger):
    def __init__(self, n_completed_steps, n_total_steps):
        super().__init__(count_mode='steps')
        self.n_completed_steps = n_completed_steps
        self.n_total_steps = n_total_steps

    def on_train_begin(self, logs=None):
        super(ProgbarCallback, self).on_train_begin(logs)
        self.seen = self.n_completed_steps
        self.target = self.n_total_steps
        self._maybe_init_progbar()

    def on_epoch_begin(self, epoch, logs=None):
        # We do not use epochs as a concept.
        self.step = epoch
        pass

    def on_train_batch_end(self, batch, logs=None):
        batch = self.step
        if logs is not None:
            # Progbar should not average anything, we want to see raw info
            self.progbar._update_stateful_metrics(list(logs.keys()))
        super(ProgbarCallback, self).on_train_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        # We do not use epochs as a concept.
        pass

    def on_train_end(self, logs=None):
        super(ProgbarCallback, self).on_epoch_end(logs)
        super(ProgbarCallback, self).on_train_end(logs)


class MyProgbar(keras.callbacks.Progbar):
    """Modified keras progbar to support starting (restoring) at an arbitrary step."""

    def __init__(self,
                 target,
                 width=30,
                 verbose=1,
                 interval=0.05,
                 stateful_metrics=None,
                 unit_name='step'):
        super(MyProgbar, self).__init__(
            target, width, verbose, interval, stateful_metrics, unit_name)
        self._initial_step = None

    def _estimate_step_duration(self, current, now):
        if self._initial_step is None:
            self._initial_step = current - 1

        if current:
            # Modified this to take into account the _initial_step
            if self._time_after_first_step is not None and current > self._initial_step + 1:
                time_per_unit = (
                        (now - self._time_after_first_step) / (current - (self._initial_step + 1)))
            else:
                time_per_unit = (now - self._start) / (current - self._initial_step)

            if current == self._initial_step + 1:
                self._time_after_first_step = now
            return time_per_unit
        else:
            return 0


# Monkey patch
keras.callbacks.Progbar = MyProgbar


class SwitchToInferenceModeCallback(keras.callbacks.Callback):
    def __init__(self, global_step_var, step_to_switch_to_inference_mode):
        super().__init__()
        self.global_step_var = global_step_var
        self.step_to_switch_to_inference_mode = step_to_switch_to_inference_mode

    def on_train_batch_begin(self, batch, logs=None):
        if (self.global_step_var > self.step_to_switch_to_inference_mode
                and not self.model.train_in_inference_mode):
            self.model.train_in_inference_mode = True
            self.model.make_train_function(force=True)
