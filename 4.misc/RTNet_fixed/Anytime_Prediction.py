import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from PIL import Image
from torch.utils.data.dataset import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Check Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 500

AlexTransform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist-data/', train=True, download=True, transform=AlexTransform),
        batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist-data/', train=False, transform=AlexTransform),
        batch_size=batch_size, shuffle=False)


# AlexNet
class alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fc1 = nn.Linear(27*27*96, 10)
        self.fc2 = nn.Linear(13*13*256, 10)
        self.fc3 = nn.Linear(13*13*384, 10)
        self.fc4 = nn.Linear(13*13*384, 10)
        self.fc5 = nn.Linear(256 * 6 * 6, 4096)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, 10)

    def forward(self, x):
        out = self.conv1(x)
        out1 = out.view(out.size(0), -1)
        out1 = self.fc1(out1)
        out1 = F.log_softmax(out1, dim=1)

        out = self.conv2(out)
        out2 = out.view(out.size(0), -1)
        out2 = self.fc2(out2)
        out2 = F.log_softmax(out2, dim=1)

        out = self.conv3(out)
        out3 = out.view(out.size(0), -1)
        out3 = self.fc3(out3)
        out3 = F.log_softmax(out3, dim=1)

        out = self.conv4(out)
        out4 = out.view(out.size(0), -1)
        out4 = self.fc4(out4)
        out4 = F.log_softmax(out4, dim=1)

        out = self.conv5(out)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc5(out))  # 256*6*6 -> 4096
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc6(out))
        out = F.dropout(out, 0.5)
        out = self.fc7(out)
        out = F.log_softmax(out, dim=1)

        return out1, out2, out3, out4, out

cnn = alexnet().to(device)
optimizer = optim.Adam(cnn.parameters())

def compute_loss(out1, out2, out3, out4, out5, y):
    loss_func = nn.CrossEntropyLoss()
    loss1 = loss_func(out1, y)
    loss2 = loss_func(out2, y)
    loss3 = loss_func(out3, y)
    loss4 = loss_func(out4, y)
    loss5 = loss_func(out5, y)
    loss = 1 * loss1 + 2 * loss2 + 3 * loss3 + 4 * loss4 + 5 * loss5
    return loss


def train(cnn, train_loader, epoch, num_epochs):
    cnn.train()

    # Train the model
    total_step = len(train_loader)

    for i, (images, labels) in enumerate(train_loader):
        # gives batch data, normalize x when iterate train_loader
        b_x = Variable(images).to(device)  # batch x
        b_y = Variable(labels).to(device)  # batch y
        output1, output2, output3, output4, output5 = cnn(b_x)
        loss = compute_loss(output1, output2, output3, output4, output5, b_y)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, i + 1, total_step, loss.item()))

# train(cnn, train_loader)

def test(noise=0):

    # Test the model
    cnn.eval()

    with torch.no_grad():

        correct = 0
        total = 0
        i = 0
        soft_out1 = torch.zeros((10000,10))
        soft_out2 = torch.zeros((10000,10))
        soft_out3 = torch.zeros((10000,10))
        soft_out4 = torch.zeros((10000,10))
        soft_out5 = torch.zeros((10000,10))

        for images, labels in test_loader:
            images = images + noise * torch.rand(images.shape)
            test_output1, test_output2, test_output3, test_output4, test_output5 = cnn(images.to(device))
            soft_out1[i*batch_size:(i+1)*batch_size,:] = test_output1
            soft_out2[i*batch_size:(i+1)*batch_size,:] = test_output2
            soft_out3[i*batch_size:(i+1)*batch_size,:] = test_output3
            soft_out4[i*batch_size:(i+1)*batch_size,:] = test_output4
            soft_out5[i*batch_size:(i+1)*batch_size,:] = test_output5
            pred_y1 = torch.max(test_output1, 1)[1].data.squeeze()
            pred_y2 = torch.max(test_output2, 1)[1].data.squeeze()
            pred_y3 = torch.max(test_output3, 1)[1].data.squeeze()
            pred_y4 = torch.max(test_output4, 1)[1].data.squeeze()
            pred_y5 = torch.max(test_output5, 1)[1].data.squeeze()
            accuracy1 = (pred_y1 == labels.to(device)).sum().item() / float(labels.to(device).size(0))
            accuracy2 = (pred_y2 == labels.to(device)).sum().item() / float(labels.to(device).size(0))
            accuracy3 = (pred_y3 == labels.to(device)).sum().item() / float(labels.to(device).size(0))
            accuracy4 = (pred_y4 == labels.to(device)).sum().item() / float(labels.to(device).size(0))
            accuracy5 = (pred_y5 == labels.to(device)).sum().item() / float(labels.to(device).size(0))

            i = i + 1

        print('Test Accuracy of the layer1 on the 10000 test images: %.2f' % accuracy1)
        print('Test Accuracy of the layer2 on the 10000 test images: %.2f' % accuracy2)
        print('Test Accuracy of the layer3 on the 10000 test images: %.2f' % accuracy3)
        print('Test Accuracy of the layer4 on the 10000 test images: %.2f' % accuracy4)
        print('Test Accuracy of the layer5 on the 10000 test images: %.2f' % accuracy5)

    return soft_out1.cpu().numpy(), soft_out2.cpu().numpy(), soft_out3.cpu().numpy(), soft_out4.cpu().numpy(), soft_out5.cpu().numpy()


load_path = 'LOAD'
save_path = 'Any_save'
noise_level = [2, 3]
# for i in range(61):
for i in range(5):
    model_num = str(i+1).zfill(2)
    cnn.load_state_dict(torch.load(load_path + '/model_' + model_num))
    print('################### model ' + model_num + ' ###################')
    anytime_df = pd.DataFrame(data=[])
    for noise in noise_level:
        out1, out2, out3, out4, out5 = np.exp(test(noise)) # converting log softmax into softmax by exp
        out_dict = {'noise': noise,
                    'Layer1 resp': out1.argmax(axis=1),
                    'Layer1 conf': out1.max(axis=1),
                    'Layer2 resp': out2.argmax(axis=1),
                    'Layer2 conf': out2.max(axis=1),
                    'Layer3 resp': out3.argmax(axis=1),
                    'Layer3 conf': out3.max(axis=1),
                    'Layer4 resp': out4.argmax(axis=1),
                    'Layer4 conf': out4.max(axis=1),
                    'Layer5 resp': out5.argmax(axis=1),
                    'Layer5 conf': out5.max(axis=1)}
        df = pd.DataFrame(data=out_dict)
        anytime_df = anytime_df.append(df)# uninstall pandas == 2.0.3
        # anytime_df = anytime_df.concat(df)

    anytime_df = anytime_df.reset_index()
    anytime_df.rename(columns={'index':'image_index'}, inplace=True)
    anytime_df['image_index'] = anytime_df['image_index'] + 1
    anytime_df.to_csv(save_path + '/anytime_prediction_' + model_num +'.csv')




