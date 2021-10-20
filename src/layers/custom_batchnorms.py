import keras
import tensorflow as tf

from init import FLAGS


class GhostBatchNormalization(keras.layers.BatchNormalization):
    """Splits the batch into virtual batches and normalizes them separately."""

    def __init__(self, split=1, **kwargs):
        if not kwargs.get('name', None):
            prefix = 'tpu_' if FLAGS.backbone.startswith('efficientnetv2') else ''
            kwargs['name'] = prefix + 'batch_normalization'
        super(GhostBatchNormalization, self).__init__(**kwargs)
        self.split = split

    def call(self, inputs, training=None):
        if not training:
            return super(GhostBatchNormalization, self).call(inputs, training=False)

        split_inputs = tf.split(inputs, self.split, axis=0)
        split_outputs = [super(GhostBatchNormalization, self).call(x, training=training)
                         for x in split_inputs]
        return tf.concat(split_outputs, axis=0)
