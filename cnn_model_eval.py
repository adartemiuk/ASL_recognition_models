import argparse

import numpy as np
import tensorflow_model_optimization as tfmot
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--Path")
parser.add_argument("-s", "--Shape")
parser.add_argument("-cm", "--ColorMode")
parser.add_argument("-qs", "--QuantizeScope", action='store_true')

args = parser.parse_args()

model_path = args.Path
input_shape = int(args.Shape)
color_mode = args.ColorMode
use_quantize_scope = bool(args.QuantizeScope)


def evaluate_quantized_model(path, shape):
    with tfmot.quantization.keras.quantize_scope():
        model = load_model(path)
    print("Model structure:\n {0}", model.summary())
    evaluate(model, shape)


def evaluate_model(path, shape):
    model = load_model(path)
    print("Model structure:\n {0}", model.summary())
    evaluate(model, shape)


def evaluate(model, shape):
    print("Evaluating model from path: %s" % model_path)
    test_datagen = ImageDataGenerator(dtype="uint8")

    test_generator = test_datagen.flow_from_directory('converted/test', target_size=(shape, shape),
                                                      batch_size=1,
                                                      class_mode="categorical", color_mode=color_mode, shuffle=False)

    y_true = test_generator.classes
    y_pred = model.predict(test_generator)
    score = model.evaluate(test_generator)
    print("Evaluation on test set:")
    print("%s: %.2f" % (model.metrics_names[0], score[0]))
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    evaluation = classification_report(y_true, np.argmax(y_pred, axis=1),
                                       labels=[i for i in range(37)])
    print(evaluation)

    conf_matrix = confusion_matrix(y_true, np.argmax(y_pred, axis=1), labels=[i for i in range(37)])
    print(conf_matrix)
    txt_file_name = "model_eval_confusion_matrix.txt"
    np.savetxt(txt_file_name, conf_matrix, fmt="% 4d")


if use_quantize_scope:
    evaluate_quantized_model(model_path, input_shape)
else:
    evaluate_model(model_path, input_shape)
