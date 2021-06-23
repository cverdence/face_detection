# Load libraries
import os
import matplotlib.image as mpimg
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import transform


# Create facial keypoint dataset class
class FacialKeypointsDataset(Dataset):

    def __init__(self, key_points, root_dir, transform=None):
        self.key_pts_frame = key_points
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.key_pts_frame.iloc[idx, 0])
        image = mpimg.imread(image_name)

        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        key_pts = self.key_pts_frame.iloc[idx, 1:].to_numpy()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample


# Create transformations
class Normalize(object):

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
        image_copy = image_copy/255.0
        key_pts_copy = (key_pts_copy - 100)/50.0

        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))
        key_pts = key_pts*[new_w/w, new_h/h]

        return {'image': image, 'keypoints': key_pts}


class RescaleImage(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        h, w = sample.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(sample, (new_h, new_w))

        return image


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class RandomCropImage(object): # Have to crop because the image is rescaled to the same scale, not exactly to the input

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        h, w = sample.shape[:2]
        new_h, new_w = self.output_size

        if h > new_h:
            top = np.random.randint(0, h - new_h)
        elif h == new_h:
            top = 0
        else:
            top = np.random.randint(h - new_h, 0)

        if w > new_w:
            left = np.random.randint(0, w - new_w)
        elif w == new_w:
            left = 0
        else:
            left = np.random.randint(w - new_w, 0)

        image = sample[top: top + new_h, left: left + new_w]
        return image


class ToTensor(object):

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)

        image = image.transpose(2, 0, 1)

        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}