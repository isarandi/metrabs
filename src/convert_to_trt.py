import tensorflow as tf
import sys
params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP16')
path = sys.argv[1]
converter = tf.experimental.tensorrt.Converter(
    input_saved_model_dir=path, conversion_params=params)
converter.convert()
converter.save(path + '_tensorrt')
