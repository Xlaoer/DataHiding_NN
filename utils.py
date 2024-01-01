import os
import torch
import numpy as np
import math
import cv2
from torchvision.utils import save_image


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
def make_dir(time):
    train_dir = 'results/{}/train_process'.format(time)

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 256, 256)
    save_image(img, name)

def calculate_psnr(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    image_size = [512, 512]  # 将图像转化为512*512大小的尺寸
    img1 = cv2.resize(img1, image_size, interpolation=cv2.INTER_CUBIC)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(img2, image_size, interpolation=cv2.INTER_CUBIC)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mse = np.mean((img1/1.0 - img2/1.0)**2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    return psnr
