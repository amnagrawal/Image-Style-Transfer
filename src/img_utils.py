from configparser import ConfigParser

from tensorflow.python.keras.preprocessing import image as tf_img
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt

config_file = "./config.ini"

config = ConfigParser()
config.read(config_file)


def load_img(path):
    img = Image.open(path)
    height = int(config.get('img_size', 'height'))
    width = int(config.get('img_size', 'width'))
    img = img.resize((height, width))

    img = tf_img.img_to_array(img)

    # We need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def invert_img(processed_img):
    x = processed_img.copy()
    # invert the preprocessing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def preprocess_img(img):
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def display_img(img, title=None):
    img = np.squeeze(img, axis=0)
    img = img.astype('uint8')
    b, g, r = cv2.split(img)  # get b,g,r
    img = cv2.merge([r, g, b])
    if title is not None:
        plt.title(title)
    plt.imshow(img)


def load_processed_img(path):
    img = load_img(path)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img
