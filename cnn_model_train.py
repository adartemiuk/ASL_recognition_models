import argparse
import datetime
import os
from glob import glob

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras import backend as K
from keras.applications import InceptionV3
from keras.applications import VGG19
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

from distiller import Distiller

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--Epochs")
parser.add_argument("-bs", "--BatchSize")
parser.add_argument("-m", "--Model")
parser.add_argument("-s", "--InputShape")
parser.add_argument("-cm", "--ColorMode")
args = parser.parse_args()

epochs = int(args.Epochs)
batch_size = int(args.BatchSize)
selected_model = args.Model
image_x = image_y = int(args.InputShape)
color_mode = args.ColorMode

K.set_image_data_format('channels_last')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_num_of_classes():
    return len(glob('gestures/*'))


def train_inception_v3_model(train_generator, num_of_train_images, batch_size, epochs, val_generator,
                             num_of_val_images):
    num_of_classes = get_num_of_classes()
    inception_model = InceptionV3(weights='imagenet', input_shape=(image_x, image_y, 3), include_top=False)

    x = inception_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_of_classes, activation='softmax')(x)

    model = Model(inputs=inception_model.input, outputs=output)
    inception_model.trainable = True

    sgd = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    filepath = "cnn_model_keras_inception_v3_checkpoint.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
    log_dir = "logs/fit/inception_v3_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [checkpoint, early_stop, tensorboard_callback]

    model.summary()

    # epochs=50, batch=24
    model.fit(train_generator, steps_per_epoch=num_of_train_images // batch_size, epochs=epochs,
              validation_data=val_generator, validation_steps=num_of_val_images // batch_size,
              callbacks=callbacks_list)
    model.save('cnn_model_inception_v3.h5')


def train_vgg_19_model(train_generator, num_of_train_images, batch_size, epochs, val_generator, num_of_val_images):
    num_of_classes = get_num_of_classes()
    vgg_model = VGG19(weights='imagenet', input_shape=(image_x, image_y, 3), include_top=False)
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(84, activation='relu')(x)
    output = Dense(num_of_classes, activation='softmax')(x)

    model = Model(inputs=vgg_model.input, outputs=output)
    for layer in vgg_model.layers[:-5]:
        layer.trainable = False
    for layer in vgg_model.layers[-5:]:
        layer.trainable = True

    sgd = SGD(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    filepath = "cnn_model_keras_vgg_19_checkpoint.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
    log_dir = "logs/fit/vgg_19_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [checkpoint, early_stop, tensorboard_callback]

    model.summary()
    # 100 epochs, 64 batch size
    model.fit(train_generator, steps_per_epoch=num_of_train_images // batch_size, epochs=epochs,
              validation_data=val_generator, validation_steps=num_of_val_images // batch_size,
              callbacks=callbacks_list)
    model.save('cnn_model_vgg_19.h5')


def train_custom_model_from_paper(train_generator, num_of_train_images, batch_size, epochs, val_generator,
                                  num_of_val_images):
    num_of_classes = get_num_of_classes()
    model = Sequential()
    # 1st layer/input
    model.add(Conv2D(32, (3, 3), input_shape=(image_x, image_y, 3), activation='leaky_relu', padding='same'))
    # 1st pool
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 2nd layer
    model.add(Conv2D(64, (3, 3), activation='leaky_relu', padding='same'))
    # 2nd pool
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 3rd layer
    model.add(Conv2D(128, (3, 3), activation='leaky_relu', padding='same'))
    # 3rd pool
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 4th layer
    model.add(Conv2D(256, (3, 3), activation='leaky_relu', padding='same'))
    # 4th pool
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 5th layer
    model.add(Conv2D(512, (3, 3), activation='leaky_relu', padding='same'))
    # 5th pool
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # flatten
    model.add(Flatten())
    # dense
    model.add(Dense(128, activation='leaky_relu'))
    # Output
    model.add(Dense(num_of_classes, activation='softmax'))

    adam = Adam(learning_rate=0.0001)

    lr_scheduler = LearningRateScheduler(schedule=lr_step_decay)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    filepath = "cnn_model_keras_custom_checkpoint.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
    log_dir = "logs/fit/custom_model_from_paper/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [checkpoint, early_stop, tensorboard_callback, lr_scheduler]

    model.summary()

    # epochs=50, batch=64
    model.fit(train_generator, steps_per_epoch=num_of_train_images // batch_size, epochs=epochs,
              validation_data=val_generator, validation_steps=num_of_val_images // batch_size,
              callbacks=callbacks_list)
    model.save('cnn_model_custom.h5')


def lr_step_decay(epoch, lr):
    if epoch > 0 and epoch % 10 == 0:
        return lr / 10
    return lr


def get_base_model():
    num_of_classes = get_num_of_classes()
    model = Sequential()
    # 1st layer/input
    model.add(
        Conv2D(96, (11, 11), input_shape=(image_x, image_y, 1), strides=(2, 2), activation='relu', padding='same'))
    # 1st pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 2nd layer
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    # 3rd layer
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    # 2nd pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 4th layer
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    # 5th layer
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    # 6th layer
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    # 3rd pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 7th layer
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    # 8th layer
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    # 9th layer
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    # 4th pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 10th layer
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # 11th layer
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # 12th layer
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # 5th pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 13th layer
    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    # 14th layer
    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    # 15th layer
    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    # 6th pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # flatten
    model.add(Flatten())
    # dense
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    # Output
    model.add(Dense(num_of_classes, activation='softmax'))

    return model


def get_student_model():
    num_of_classes = get_num_of_classes()
    model = Sequential()
    # 1st layer/input
    model.add(
        Conv2D(96, (11, 11), input_shape=(image_x, image_y, 1), strides=(2, 2), activation='relu', padding='same'))
    # 1st pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 2nd layer
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    # 3rd layer
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    # 2nd pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 4th layer
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # 3rd pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 5th layer
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # 4th pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 6th layer
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    # 5th pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 7th layer
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # 6th pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # flatten
    model.add(Flatten())
    # dense
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    # Output
    model.add(Dense(num_of_classes, activation='softmax'))

    return model


def get_model_with_layer_decomposition():
    num_of_classes = get_num_of_classes()
    model = Sequential()
    # 1st layer/input
    model.add(
        Conv2D(96, (11, 11), input_shape=(image_x, image_y, 1), strides=(2, 2), activation='relu', padding='same'))
    # 1st pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 2nd layer
    model.add(SeparableConv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    # 3rd layer
    model.add(SeparableConv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    # 2nd pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 4th layer
    model.add(SeparableConv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    # 5th layer
    model.add(SeparableConv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    # 6th layer
    model.add(SeparableConv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    # 3rd pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 7th layer
    model.add(SeparableConv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    # 8th layer
    model.add(SeparableConv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    # 9th layer
    model.add(SeparableConv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    # 4th pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 10th layer
    model.add(SeparableConv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    # 11th layer
    model.add(SeparableConv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    # 12th layer
    model.add(SeparableConv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    # 5th pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 13th layer
    model.add(SeparableConv2D(1024, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    # 14th layer
    model.add(SeparableConv2D(1024, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    # 15th layer
    model.add(SeparableConv2D(1024, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    # 6th pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # flatten
    model.add(Flatten())
    # dense
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    # Output
    model.add(Dense(num_of_classes, activation='softmax'))

    return model


def set_up_decomposed_model(train_generator, num_of_train_images, batch_size, epochs, val_generator, num_of_val_images):
    model = get_model_with_layer_decomposition()
    sgd = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    filepath = "cnn_model_keras_decomposed_checkpoint.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
    log_dir = "logs/fit/decomposed_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [checkpoint, early_stop, tensorboard_callback]

    model.summary()
    model.fit(train_generator, steps_per_epoch=num_of_train_images // batch_size, epochs=epochs,
              validation_data=val_generator, validation_steps=num_of_val_images // batch_size,
              callbacks=callbacks_list)
    model.save('cnn_model_decomposed.h5')


# QAT
def set_up_qat_model(train_generator, num_of_train_images, batch_size, epochs, val_generator, num_of_val_images):
    model = get_base_model()

    sgd = Adam(learning_rate=0.0001)
    quantize_model = tfmot.quantization.keras.quantize_model
    q_model = quantize_model(model)
    q_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    filepath = "cnn_model_keras_qat_checkpoint.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
    log_dir = "logs/fit/qat_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [checkpoint, early_stop, tensorboard_callback]

    model.summary()
    model.fit(train_generator, steps_per_epoch=num_of_train_images // batch_size, epochs=epochs,
              validation_data=val_generator, validation_steps=num_of_val_images // batch_size,
              callbacks=callbacks_list)
    model.save('cnn_model_qat.h5')


def set_up_knowledge_distillation_model(model_path, train_generator, num_of_train_images, batch_size, epochs,
                                        val_generator, num_of_val_images):
    teacher_model = tf.keras.models.load_model(model_path)
    student_model = get_student_model()
    print("Teacher: ")
    teacher_model.summary()
    print(3 * "\n")
    print("Student: ")
    student_model.summary()
    # Initialize and compile distiller
    distiller = Distiller(student=student_model, teacher=teacher_model)
    distiller.compile(
        optimizer=Adam(learning_rate=0.0001),
        metrics=[keras.metrics.CategoricalAccuracy()],
        student_loss_fn=keras.losses.CategoricalCrossentropy(),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )
    # early_stop = EarlyStopping(monitor='val_student_loss', verbose=1, patience=5)
    log_dir = "logs/fit/knowledge_distillation_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [tensorboard_callback]

    distiller.fit(train_generator, steps_per_epoch=num_of_train_images // batch_size, epochs=epochs,
                  validation_data=val_generator, validation_steps=num_of_val_images // batch_size,
                  callbacks=callbacks_list)

    student_model.compile(loss='categorical_crossentropy', metrics=["accuracy"])

    student_model.save('cnn_model_from_knowledge_distillation.h5')


def set_up_base_model(train_generator, num_of_train_images, batch_size, epochs, val_generator, num_of_val_images):
    model = get_base_model()
    sgd = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    filepath = "cnn_model_base_checkpoint.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
    log_dir = "logs/fit/base_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [checkpoint, early_stop, tensorboard_callback]

    model.summary()
    model.fit(train_generator, steps_per_epoch=num_of_train_images // batch_size, epochs=epochs,
              validation_data=val_generator, validation_steps=num_of_val_images // batch_size,
              callbacks=callbacks_list)
    model.save('cnn_model_base.h5')


def train(batch_size, epochs, selected_model, color_mode):
    num_of_train_images = len(glob('converted/train/**/*.jpg', recursive=True))
    num_of_val_images = len(glob('converted/val/**/*.jpg', recursive=True))
    train_datagen = ImageDataGenerator(dtype="uint8")
    val_datagen = ImageDataGenerator(dtype="uint8")

    train_generator = train_datagen.flow_from_directory('converted/train', target_size=(image_x, image_y),
                                                        batch_size=batch_size,
                                                        class_mode="categorical", color_mode=color_mode)
    val_generator = val_datagen.flow_from_directory('converted/val', target_size=(image_x, image_y),
                                                    batch_size=batch_size,
                                                    class_mode="categorical", color_mode=color_mode)
    print("Is GPU Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    if selected_model == "BASE":
        print("INFO - BASE MODEL TRAINING")
        set_up_base_model(train_generator, num_of_train_images, batch_size, epochs, val_generator, num_of_val_images)
    elif selected_model == "QAT":
        print("INFO - QAT OPTIMIZED MODEL TRAINING")
        set_up_qat_model(train_generator, num_of_train_images, batch_size, epochs, val_generator, num_of_val_images)
    elif selected_model == "LAYER_DECOMPOSITION":
        print("INFO - LAYER_DECOMPOSITION OPTIMIZED MODEL TRAINING")
        set_up_decomposed_model(train_generator, num_of_train_images, batch_size, epochs, val_generator,
                                num_of_val_images)
    elif selected_model == "KNOWLEDGE_DISTILLATION":
        print("INFO - KNOWLEDGE_DISTILLATION OPTIMIZED MODEL TRAINING")
        set_up_knowledge_distillation_model("base_model\\cnn_model_base.h5", train_generator, num_of_train_images,
                                            batch_size, epochs, val_generator, num_of_val_images)
    elif selected_model == "INCEPTION":
        print("INFO - INCEPTION V3 MODEL TRAINING")
        train_inception_v3_model(train_generator, num_of_train_images, batch_size, epochs, val_generator,
                                 num_of_val_images)
    elif selected_model == "VGG19":
        print("INFO - VGG19 MODEL TRAINING")
        train_vgg_19_model(train_generator, num_of_train_images, batch_size, epochs, val_generator,
                           num_of_val_images)
    elif selected_model == "TEST":
        print("INFO - TEST MODEL TRAINING")
        train_custom_model_from_paper(train_generator, num_of_train_images, batch_size, epochs, val_generator,
                                      num_of_val_images)
    else:
        print("INFO - GIVEN MODEL NOT FOUND. BASE MODEL TRAINING")
        set_up_base_model(train_generator, num_of_train_images, batch_size, epochs, val_generator, num_of_val_images)


train(batch_size, epochs, selected_model, color_mode)
K.clear_session()
