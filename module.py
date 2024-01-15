import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):
       x = self.encoder(x)
       return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # 输出范围在[0, 1]
        )
    def forward(self, x):
       x = self.decoder(x)
       x = x * 255.0
       return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.flatten = nn.Flatten()

    def forward(self, x):
        encoded = self.encoder(x)
        mid = self.flatten(encoded)
        test = mid.view(-1, 256, 32, 32)
        decoded = self.decoder(test)
        return decoded