# Load libraries
import torch
import torch.nn as nn
from facial_detection.model import ConvolutionalNeuralNetwork
from torch.utils.data import DataLoader
import facial_detection.evaluation_and_visualize as ev
import torch.optim as optim
import matplotlib.pyplot as plt


# Load data sets
transformed_dataset_train = torch.load('facial_detection/transformed_data/transformed_train_dataset.torch')
transformed_dataset_test = torch.load('facial_detection/transformed_data/transformed_test_dataset.torch')
net = ConvolutionalNeuralNetwork()

# Prepare data sets for training
batch_size = 32
train_loader = DataLoader(transformed_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(transformed_dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)

train_images, train_outputs, gt_pts = ev.net_sample_output(train_loader, net)
print(train_images.data.size(), train_outputs.data.size(), gt_pts.size())
ev.visualize_output(5, train_images, train_outputs, gt_pts)

test_images, test_outputs, gt_pts = ev.net_sample_output(test_loader, net)
print(test_images.data.size(), test_outputs.data.size(), gt_pts.size())
ev.visualize_output(5, test_images, test_outputs, gt_pts)

# Define loss and optimization
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train
n_epochs = 1
loss = net.train_net(n_epochs, train_loader, optimizer, criterion)
plt.plot(loss)

# Save model
torch.save(net.state_dict(), 'facial_detection/saved_models/trained_model_2.pt')

# Feature visualization
weights1 = net.conv1.weight.data
w = weights1.numpy()
filter_index = 0
print(w[filter_index][0])
print(w[filter_index][0].shape)

plt.imshow(w[filter_index][0], cmap='gray')