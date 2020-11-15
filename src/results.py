from matplotlib import pyplot as plt
import numpy as np
from img_utils import display_img


def plot_performance(losses, title=None):
    plt.figure(figsize=(5, 5))
    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    if title is not None:
        plt.title(title)


def plot_images(images, display_interval):
    num_rows = 2
    num_cols = len(images)/num_rows
    plt.figure(figsize=(20, 10))

    for i, img in enumerate(images):
        plt.subplot(num_rows, num_cols, i+1)
        plt.title(f'Iteration: {(i + 1) * display_interval}')
        # img = np.squeeze(img, axis=0).astype('uint8')
        plt.imshow(img)
        # display_img(img)