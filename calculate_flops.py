import argparse

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--Path")
parser.add_argument("-qs", "--QuantizeScope", action='store_true')
args = parser.parse_args()

model_path = args.Path
use_quantize_scope = bool(args.QuantizeScope)


def get_flops_for_model():
    if use_quantize_scope:
        print("INFO - Using QUANTIZE SCOPE")
        with tfmot.quantization.keras.quantize_scope():
            model = tf.keras.models.load_model(model_path)

    else:
        print("INFO - Using non QUANTIZE SCOPE")
        model = tf.keras.models.load_model(model_path)

    input = tf.TensorSpec([1] + model.inputs[0].shape[1:], model.inputs[0].dtype)

    model_fun = tf.function(model).get_concrete_function(input)

    frozen_model_fun = convert_variables_to_constants_v2(model_fun)

    run_meta_data = tf.compat.v1.RunMetadata()

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_model_fun.graph, run_meta=run_meta_data, cmd='op', options=opts)

    return flops


flops = get_flops_for_model()
print(100 * '-')
print("FLOPs (M): {}".format(float(flops.total_float_ops) * 1e-6))
print(100 * '-')
print("MACs (M): {}".format(float(flops.total_float_ops // 2) * 1e-6))
