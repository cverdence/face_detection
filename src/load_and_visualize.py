# Load libraries
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Load all images from folder
def load_images_from_folder(folder, color):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            if color == "rgb":
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif color == "gray":
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            else:
                raise Exception('Color should be rgb or gray. The value of color was: {}'.format(color))
    return images


# Plot images with keypoints
def plot_images_from_dir_with_keypoints(number_of_images_height, number_of_images_width, image_dir, key_points):
    fig, axs = plt.subplots(number_of_images_width, number_of_images_height,
                            figsize=(15, 8), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.2)
    axs = axs.ravel()

    for i in range(number_of_images_width*number_of_images_height):
        image_name = key_points.iloc[i, 0]
        key_pts = key_points.iloc[i, 1:].to_numpy()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        axs[i].scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')
        axs[i].imshow(mpimg.imread(os.path.join(image_dir, image_name)))
        axs[i].set_title('Picture: ' + image_name)


def show_keypoints(image, key_pts, color=None):
    if color == 'gray':
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')


def plot_images_with_keypoints(number_of_images_height, number_of_images_width, images, key_points):
    fig, axs = plt.subplots(number_of_images_width, number_of_images_height,
                            figsize=(15, 8), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.2)
    axs = axs.ravel()

    for i in range(number_of_images_width*number_of_images_height):
        axs[i].imshow(images[i])
        axs[i].scatter(key_points[i][:, 0], key_points[i][:, 1], s=20, marker='.', c='m')
        axs[i].set_title('Picture: ' + str(i))
