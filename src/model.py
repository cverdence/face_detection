# Load libraries
import torch.nn as nn
import torch


# Create convolutional neural network class
class ConvolutionalNeuralNetwork(nn.Module):

    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*53*53, 244)
        self.fc1_drop = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(244, 136)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        return x

    def train_net(self, n_epochs, loader, optimizer, criterion):
        self.train()
        loss_development_batch = []
        loss_development_epoch = []

        for epoch in range(n_epochs):

            running_loss = 0.0

            for batch_i, data in enumerate(loader):
                images = data['image']
                key_pts = data['keypoints']
                key_pts = key_pts.view(key_pts.size(0), -1)
                key_pts = key_pts.type(torch.FloatTensor)
                images = images.type(torch.FloatTensor)
                output_pts = self(images)
                loss = criterion(output_pts, key_pts)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loss_development_batch.append(loss.item())
                if batch_i % 10 == 9:
                    print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 10))

            loss_development_epoch.append(running_loss)

        print('Finished Training')
        return loss_development_epoch