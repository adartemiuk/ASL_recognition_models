import argparse

import tensorflow as tf
import tensorflow_model_optimization as tfmot

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--Input")
parser.add_argument("-o", "--Output")
parser.add_argument("-qs", "--QuantizeScope", action='store_true')
parser.add_argument("-q", "--Quantize", action='store_true')
args = parser.parse_args()

model_path = args.Input
tflite_model_output_path = args.Output
use_quantize_scope = bool(args.QuantizeScope)
quantize = bool(args.Quantize)

if use_quantize_scope:
    print("INFO - Using QUANTIZE SCOPE")
    with tfmot.quantization.keras.quantize_scope():
        model = tf.keras.models.load_model(model_path)
else:
    print("INFO - Using non QUANTIZE SCOPE")
    model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

if quantize:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    print("INFO - Converting model to tflite with quantization")

else:
    print("INFO - Converting model to tflite without quantization")

tfm_model = converter.convert()
open(tflite_model_output_path, "wb").write(tfm_model)
print(str("INFO - {} tflite model saved").format("QUANTIZED" if quantize else "NORMAL"))
