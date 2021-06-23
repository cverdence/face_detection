# Load libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from src.model import ConvolutionalNeuralNetwork
import src.transforms as t
import src.evaluate_and_visualize as ev

# Plot image
image = cv2.imread('data/images/obamas.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)
fig = plt.figure(figsize=(9, 6))
plt.imshow(image)

# Detect faces
face_cascade = cv2.CascadeClassifier('data/detector/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(image, 1.2, 2)
image_with_detections = image.copy()

for (x, y, w, h) in faces:
    cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (255, 0, 0), 3)

fig = plt.figure(figsize=(9, 6))
plt.imshow(image_with_detections)

# Test model
net = ConvolutionalNeuralNetwork()
net.load_state_dict(torch.load('data/saved_models/trained_model.pt'))
net.eval()

# Insert predicted keypoints
image_copy = np.copy(image)
rescale_image = t.RescaleImage(224)
random_crop = t.RandomCropImage(224)
padding = 75

for i, (x, y, w, h) in enumerate(faces):
    roi = image_copy[y - padding:y + h + padding, x - padding:x + w + padding]
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    roi = roi/255.0
    roi = rescale_image(roi)
    roi = random_crop(roi)
    roi = torch.from_numpy(roi)
    if i == 0:
        images = roi
    else:
        images = torch.cat((images, roi), 0)

images = images.resize(2, 224, 224)
images = images.unsqueeze(1)
images = images.type(torch.FloatTensor)
key_pts = net.forward(images)
key_pts = key_pts.resize(2, 68, 2)
images = images.resize(2, 224, 224)

# Visualize
ev.visualize_output(1, images, key_pts, transposed=True)