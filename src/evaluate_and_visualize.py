# Load libraries
import torch
import matplotlib.pyplot as plt
import numpy as np


# Output sample
def net_sample_output(loader, net):
    for i, sample in enumerate(loader):
        images = sample['image']
        key_pts = sample['keypoints']
        images = images.type(torch.FloatTensor)
        output_pts = net.forward(images)
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        if i == 0:
            return images, output_pts, key_pts


# Visualize
def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    if gt_pts is not None:
        print('yes')
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


def visualize_output(n_img_half, images, outputs, gt_pts=None, transposed=False):
    fig, axs = plt.subplots(2, n_img_half, figsize=(15, 8), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.2)
    axs = axs.ravel()

    for i in range(n_img_half*2):
        image = images[i].data
        image = image.numpy()
        if not transposed:
            image = np.transpose(image, (1, 2, 0))

        predicted_key_pts = outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        predicted_key_pts = predicted_key_pts * 50.0 + 100

        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts * 50.0 + 100

        axs[i].imshow(image, cmap='gray')
        axs[i].scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
        if gt_pts is not None:
            axs[i].scatter(ground_truth_pts[:, 0], ground_truth_pts[:, 1], s=20, marker='.', c='g')

        plt.axis('off')


def plot_multiple_images(height, width, rgb_images):
    fig, axs = plt.subplots(height, width, figsize=(15, 8), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.2)
    axs = axs.ravel()
    for i in range(width*height):
        image = rgb_images[i]
        axs[i].imshow(image)