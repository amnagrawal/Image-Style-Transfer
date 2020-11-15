import os
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2

from features import get_feature_representations
from img_utils import load_img, display_img, invert_img, load_processed_img
from loss import gram_features
from model import compute_grads, get_model
from results import plot_images, plot_performance

data_dir = "../data/"
config_file = "./config.ini"

config = ConfigParser()
config.read(config_file)
content_img = config.get('filenames', 'content_img')
style_img = config.get('filenames', 'style_img')
content_path = os.path.join(data_dir, content_img)
style_path = os.path.join(data_dir, style_img)

content = load_img(content_path).astype('uint8')
style = load_img(style_path).astype('uint8')

style_layers = config.get('layers', 'style_layers').split()
content_layers = config.get('layers', 'content_layer').split()

num_iterations = int(config.get('params', 'n_iterations'))

model = get_model()
for layer in model.layers:
    layer.trainable = False

style_features, content_features = get_feature_representations(model, content_path, style_path)
gram_style_features = gram_features(style_features)

init_image = load_processed_img(content_path)
init_image = tf.Variable(init_image, dtype=tf.float32)
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=5)

best_loss, best_img = float('inf'), None

display_interval = int(config.get('params', 'disp_interval'))

norm_means = np.array([103.939, 116.779, 123.68])
min_vals = -norm_means
max_vals = 255 - norm_means

images = []
losses = []
style_losses = []
content_losses = []
for i in range(num_iterations+1):
    grads, loss, style_loss, content_loss = compute_grads(model, init_image,
                                                          gram_style_features, content_features)
    losses.append(loss)
    style_losses.append(style_loss)
    content_losses.append(content_loss)
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)

    if loss < best_loss:
        # Update best loss and best image from total loss.
        best_loss = loss
        best_img = invert_img(init_image.numpy())

    cv2.namedWindow("iterations")
    temp = np.squeeze(best_img.copy())
    cv2.setWindowTitle("iterations", f"Iteration: {i}")
    cv2.imshow("iterations", temp)
    cv2.waitKey(10)

    if (i+1) % display_interval == 0:

        plot_img = init_image.numpy()
        plot_img = invert_img(plot_img)
        plot_img = np.squeeze(plot_img, axis=0)
        b, g, r = cv2.split(plot_img)  # get b,g,r
        plot_img = cv2.merge([r, g, b])
        images.append(plot_img)

        print("============================================")
        print(f'Iteration: {i+1}')
        print(f'Total loss: {loss:0.3f},')
        print(f'style loss: {style_loss:0.3f}, ', end="\t")
        print(f'content loss: {content_loss:0.3f}')

cv2.waitKey(0)
cv2.destroyAllWindows()

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
display_img(content, 'Content Image')

plt.subplot(1, 2, 2)
display_img(style, 'Style Image')
plt.show()

plot_performance(losses, 'Total_loss')
plot_performance(style_losses, 'Style_loss')
plot_performance(content_losses, 'content_loss')
plot_images(images, display_interval)

cv2.namedWindow("Result")
image_x = lambda x: cv2.imshow("Result", cv2.cvtColor(images[x], cv2.COLOR_RGB2BGR))
cv2.createTrackbar("Image", "Result", 0, len(images)-1, image_x)
cv2.imshow("Result", np.squeeze(best_img, axis=0).astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


