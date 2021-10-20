import abc

import keras
import keras.metrics
import tensorflow as tf
from attrdict import AttrDict


class ModelTrainer(keras.Model, metaclass=abc.ABCMeta):
    def __init__(self, global_step):
        super().__init__()
        self.global_step = global_step
        self.train_in_inference_mode = False
        self.my_metrics = {}
        self.predict_tensor_names = None

    def train_step(self, inps):
        with tf.GradientTape() as tape:
            preds = self._forward_train(inps, training=not self.train_in_inference_mode)
            losses = self._compute_losses(inps, preds)

        self.optimizer.minimize(losses['loss'], self.trainable_variables, tape=tape)
        self.global_step.assign_add(1)
        metrics = self._compute_metrics(inps, preds)
        metrics.update(losses)
        metrics['learning_rate'] = self.optimizer.learning_rate
        return {f'metrics/{k}': v for k, v in metrics.items()}

    def test_step(self, inps):
        preds = self._forward_test(inps)
        current_metrics = self._compute_metrics(inps, preds)
        for metric_name, metric_value in current_metrics.items():
            if metric_name not in self.my_metrics:
                self.my_metrics[metric_name] = keras.metrics.Mean(name=metric_name)
            self.my_metrics[metric_name].update_state(metric_value)
        return {f'metrics/{k}': v.result() for k, v in self.my_metrics.items()}

    def predict_step(self, inps):
        preds = self._forward_test(inps)
        tensors = {**inps, **preds}
        if self.predict_tensor_names is None:
            return tensors
        else:
            return {k: tensors[k] for k in self.predict_tensor_names if k in tensors}

    def reset_metrics(self):
        super().reset_metrics()
        for m in self.my_metrics.values():
            m.reset_state()

    @tf.function
    def _forward_train(self, inps, training):
        inps = AttrDict(inps)
        result = self.forward_train(inps, training)
        result['_keras_loss'] = (
            tf.add_n(self.losses) if self.losses else tf.constant(0, dtype=tf.float32))
        return dict(result)

    @tf.function
    def _forward_test(self, inps):
        inps = AttrDict(inps)
        result = self.forward_test(inps)
        return dict(result)

    @tf.function
    def _compute_losses(self, inps, preds):
        inps = AttrDict(inps)
        preds = AttrDict(preds)
        result = self.compute_losses(inps, preds)
        result['loss'] = result['loss'] + tf.cast(preds['_keras_loss'], result['loss'].dtype)
        return dict(result)

    @tf.function
    def _compute_metrics(self, inps, preds):
        inps = AttrDict(inps)
        preds = AttrDict(preds)
        result = self.compute_metrics(inps, preds)
        return dict(result)

    @abc.abstractmethod
    def forward_train(self, inps, training):
        pass

    @abc.abstractmethod
    def forward_test(self, inps):
        pass

    @abc.abstractmethod
    def compute_losses(self, inps, preds):
        pass

    @abc.abstractmethod
    def compute_metrics(self, inps, preds):
        pass

    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError('A model trainer itself should not be called.')
