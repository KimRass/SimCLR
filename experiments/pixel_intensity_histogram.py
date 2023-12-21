# References
    # https://zahid-parvez.medium.com/image-histograms-in-opencv-python-9fe3a7e0ae4f

import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_utils import load_image, _figure_to_array, save_image


def get_channel_wise_pixel_intensity_histogram(img):
    fig, axes = plt.subplots(figsize=(6, 4))
    colors = ["r","g","b"]
    for c in [0, 1, 2]:
        hist = cv2.calcHist(images=[img], channels=[c], mask=None, histSize=[256], ranges=[0, 256])
        axes.plot(hist, color=colors[c], linewidth=0.5)

    axes.set_xticks(np.arange(0, 255, 25))
    axes.tick_params(axis="x", labelrotation=90)
    axes.tick_params(axis="both", which="major", labelsize=7)
    fig.tight_layout()

    arr = _figure_to_array(fig)
    return arr


def get_pixel_intensity_histogram(img):
    fig, axes = plt.subplots(figsize=(6, 4))
    pseudo_img = np.concatenate([img[..., 0], img[..., 1], img[..., 2]], axis=0)
    hist = cv2.calcHist(images=[pseudo_img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    axes.plot(hist, linewidth=0.5, color="black")

    axes.set_xticks(np.arange(0, 255, 25))
    axes.tick_params(axis="x", labelrotation=90)
    axes.tick_params(axis="both", which="major", labelsize=7)
    fig.tight_layout()    

    arr = _figure_to_array(fig)
    return arr


if __name__ == "__main__":
    img_path = "/Users/jongbeomkim/Documents/datasets/VOCdevkit/VOC2012/JPEGImages/2012_002148.jpg"
    img = load_image(img_path)
    hist = get_channel_wise_pixel_intensity_histogram(img)
    save_image(
        hist,
        path="/Users/jongbeomkim/Desktop/workspace/visual_representation_learning/simclr/experiments/channel_wise_intensity_sample.jpg"
    )
    hist = get_pixel_intensity_histogram(img)
    save_image(
        hist,
        path="/Users/jongbeomkim/Desktop/workspace/visual_representation_learning/simclr/experiments/intensity_sample.jpg"
    )
