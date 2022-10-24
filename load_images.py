import os
from glob import glob

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array
from keras.utils import load_img


def collect_preprocessed_images(train_split_ratio, val_split_ratio, test_split_ratio):
    for dir in os.listdir("gestures"):
        get_images_regex = "gestures\\" + dir + "\\*.png"
        images = glob(get_images_regex)
        index = 0
        for image in images:
            print(image)
            preprocess_images(dir, image, index, "train", train_split_ratio)
            preprocess_images(dir, image, index, "val", val_split_ratio)
            preprocess_images(dir, image, index, "test", test_split_ratio)
            index += 1


def preprocess_images(dir, image_path, base_image_id, dataset_type, percentage_ratio):
    multiplier = 40 * percentage_ratio
    data_generator = ImageDataGenerator(width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        brightness_range=[0.2, 2.0],
                                        zoom_range=[0.5, 1.0],
                                        fill_mode='nearest')
    index = 1
    target = image_path
    img = load_img(target)
    img_arr = img_to_array(img)
    converted_image = enlarge_image(img_arr)
    samples = np.expand_dims(converted_image, 0)
    for sample in data_generator.flow(samples, batch_size=1):
        image = np.squeeze(sample, 0)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        path_to_save = f"converted_/{dataset_type}/{dir}/{dir}_{base_image_id}_{index}.jpg"
        cv2.imwrite(path_to_save, img_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        index += 1
        if index > multiplier:
            break


def enlarge_image(img_arr):
    new_img_width, new_img_height = 3 * img_arr.shape[0], 3 * img_arr.shape[1]
    x_offset, y_offset = int((new_img_width - img_arr.shape[0]) / 2), int((new_img_height - img_arr.shape[1]) / 2)
    background_img = np.zeros([new_img_width, new_img_height, 3], dtype=np.uint8)
    x1, x2 = x_offset, x_offset + img_arr.shape[0]
    y1, y2 = y_offset, y_offset + img_arr.shape[1]
    background_img[x1:x2, y1:y2] = img_arr
    return background_img
