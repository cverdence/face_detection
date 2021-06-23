# Load libraries
import src.load_and_visualize as lv
import src.transforms as t
import pandas as pd
import numpy as np
from torchvision import transforms
import torch

# Visualize images
train_dir = 'facial_detection/data/train-test-data/training'
test_dir = 'facial_detection/data/train-test-data/test'
train_key_points = pd.read_csv('facial_detection/data/train-test-data/training_frames_keypoints.csv')
test_key_points = pd.read_csv('facial_detection/data/train-test-data/test_frames_keypoints.csv')

lv.plot_images_from_dir_with_keypoints(4, 4, train_dir, train_key_points)

# Transformations
face_dataset = t.FacialKeypointsDataset(train_key_points, train_dir)

rescale = t.Rescale(100)
crop = t.RandomCrop(50)
composed = transforms.Compose([t.Rescale(250), t.RandomCrop(224)])

sample = face_dataset[300]
rescaled_sample = rescale(sample)
cropped_sample = crop(sample)
composed_sample = composed(sample)

images_transformed = np.array([sample['image'], rescaled_sample['image'], cropped_sample['image'],
                               composed_sample['image']], dtype=object)
key_pts_transformed = np.array([sample['keypoints'], rescaled_sample['keypoints'], cropped_sample['keypoints'],
                                composed_sample['keypoints']], dtype=object)

lv.plot_images_with_keypoints(2, 2, images_transformed, key_pts_transformed)

data_transform = transforms.Compose([t.Rescale(250), t.RandomCrop(224), t.Normalize(), t.ToTensor()])
transformed_train_dataset = t.FacialKeypointsDataset(train_key_points, train_dir, transform=data_transform)
transformed_test_dataset = t.FacialKeypointsDataset(test_key_points, test_dir, transform=data_transform)

# Save dataset
torch.save(transformed_train_dataset, 'data/transformed_data/transformed_train_dataset.torch')
torch.save(transformed_test_dataset, 'data/transformed_data/transformed_test_dataset.torch')