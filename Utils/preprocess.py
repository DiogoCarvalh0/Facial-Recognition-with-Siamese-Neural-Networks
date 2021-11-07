import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'

import Configs.config as config
import tensorflow as tf

def preprocess(image_path):
    byte_img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, config.INPUT_IMAGE_SIZE)
    img = img / 255.0

    return img


def preprocess_twin(input_image, validation_image, label):
    return preprocess(input_image), preprocess(validation_image), label
