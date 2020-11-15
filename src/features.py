from configparser import ConfigParser

from img_utils import load_processed_img

config_file = "./config.ini"

config = ConfigParser()
config.read(config_file)


def get_feature_representations(model, content_path, style_path):
    content_image = load_processed_img(content_path)
    style_image = load_processed_img(style_path)

    # batch compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    num_style_layers = len(config.get('layers', 'style_layers').split())

    # Get the style and content feature representations from our model
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features

