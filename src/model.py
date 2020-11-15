from configparser import ConfigParser

import tensorflow as tf
from keras import models

from loss import total_loss

config_file = "./config.ini"

config = ConfigParser()
config.read(config_file)


def get_model():
    # Load our model. We load pretrained VGG, trained on imagenet data
    style_layers = config.get('layers', 'style_layers').split()
    content_layers = config.get('layers', 'content_layer').split()
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    # Get output layers corresponding to style and content layers
    style_outputs = []
    for name in style_layers:
        style_outputs.append(vgg.get_layer(name).output)

    content_outputs = []
    for name in content_layers:
        content_outputs.append(vgg.get_layer(name).output)

    outputs = style_outputs + content_outputs
    # Build model
    return models.Model(vgg.input, outputs)


def compute_grads(model, init_image, gram_style_features, content_features):
    with tf.GradientTape() as tape:
        loss, style_loss, content_loss = total_loss(model, init_image,
                                                    gram_style_features, content_features)
    # Compute gradients wrt input image
    grad = tape.gradient(loss, init_image)
    return grad, loss, style_loss, content_loss
