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
import time
import sys
from code_final_cifar_net import *
import argparse


NUM_EPOCHS = 233
LEARNING_RATE = 1e-3
BATCH_SIZE = 12
time = time.strftime("%y%m%d_%H%M%S", time.localtime())

# net = Autoencoder()
# torch.backends.cudnn.enabled = False
parser = argparse.ArgumentParser()
# model hyper-parameters
parser.add_argument('--trans_bit', type=int, default=256)
config = parser.parse_args()
net = Our_Net(config)
print(net)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
device = get_device()
os.makedirs("./results/{}".format(time), exist_ok=True)
file = open("./results/{}/output.txt".format(time),"w")

def train(net, trainloader, NUM_EPOCHS):
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for data in trainloader:
            img = data  # no need for the labels
            img = img.to(device)
            optimizer.zero_grad()
            outputs = net(img)
            psnr_each_img = 0.0
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        info = 'Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch + 1, NUM_EPOCHS, loss)
        print(info,file=file)
        print(info)
        if epoch % 5 == 0:
            save_decoded_image(img.cpu().data, name='./results/{}/train_process/original{}.png'.format(time,epoch))
            save_decoded_image(outputs.cpu().data, name='./results/{}/train_process/decoded{}.png'.format(time,epoch))
            info = 'Epoch {} of PSNR {}'.format(epoch+1,calculate_psnr('./results/{}/train_process/original{}.png'.format(time,epoch),'./results/{}/train_process/decoded{}.png'.format(time,epoch)))
            print(info, file=file)
            print(info)
    file.close()
    return train_loss