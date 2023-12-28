import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from module import *
from utils import *

NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

net = Autoencoder()
print(net)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
device = get_device()

def train(net, trainloader, NUM_EPOCHS):
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        psnr = 0.0
        for data in trainloader:
            img, _ = data  # no need for the labels
            img = img.to(device)
            optimizer.zero_grad()
            outputs = net(img)
            psnr += calculate_psnr(img.data.numpy(),outputs.data.numpy())
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)
        psnr = psnr / len(trainloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f} , PSNR:{:.3f}'.format(
            epoch + 1, NUM_EPOCHS, loss,psnr))
        if epoch % 5 == 0:
            save_decoded_image(img.cpu().data, name='./Conv_CIFAR10_Images/original{}.png'.format(epoch))
            save_decoded_image(outputs.cpu().data, name='./Conv_CIFAR10_Images/decoded{}.png'.format(epoch))
    return train_loss