import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, hidden_dim=6):
        super(Encoder, self).__init__()
        # 编码器部分，这里使用两层卷积并跟上最大池化层来缩小特征图尺寸
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(output_dim)
        self.relu2 = nn.ReLU(True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim=1, output_dim=3, hidden_dim=6, img_size=256):
        super(Decoder, self).__init__()
        # 解码器部分，使用转置卷积（transpose convolution）将特征图恢复到原始尺寸
        self.t_conv1 = nn.ConvTranspose2d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(True)

        self.t_conv2 = nn.ConvTranspose2d(hidden_dim, output_dim, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU(True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.t_conv1(x)))
        x = self.relu2(self.t_conv2(x))
        # 输出层不使用激活函数，以便输出范围可以覆盖整个像素值区间
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)


        # int_tensor = (encoded * 255).to(torch.uint8)
        # store_carrier = np.zeros((int_tensor.size(0),1,1,1))
        # print(store_carrier)
        # for i in int_tensor:
        #     for j in i:
        #         for k in j:
        #             for m in k:
        #                 break

        decoded = self.decoder(encoded)
        return decoded