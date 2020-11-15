from configparser import ConfigParser
import sys
import tensorflow as tf

config_file = "./config.ini"

config = ConfigParser()
config.read(config_file)
num_style_layers = len(config.get('layers', 'style_layers').split())
num_content_layers = len(config.get('layers', 'content_layer').split())


def content_loss(base_content, target):
    loss = tf.reduce_mean(tf.square(base_content - target))
    return loss


def gram_matrix(tensor):
    # We make the image channels first
    num_channels = int(tensor.shape[-1])
    mat = tf.reshape(tensor, [-1, num_channels])
    n = tf.shape(mat)[0]
    gram = tf.matmul(mat, mat, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def gram_features(features):
    return [gram_matrix(feature) for feature in features]


def style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)
    m_l = len(base_style)
    n_l = 4
    # return tf.reduce_mean(tf.square(gram_style - gram_target))/(4*(m_l**2)*(n_l**2))
    return tf.reduce_mean(tf.square(gram_style - gram_target))/tf.cast((4*(m_l)*(n_l)), 'float32')


def total_loss(model, init_image, gram_style_features, content_features):
    content_weight = float(config.get('weights', 'content_weight'))
    style_weight = float(config.get('weights', 'style_weight'))

    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    weight_per_layer = style_weight / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        loss = style_loss(comb_style[0], target_style)
        loss = weight_per_layer * loss
        style_score += loss

    # Get content loss from all layers
    weight_per_layer = content_weight / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_layer * content_loss(comb_content[0], target_content)

    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score
