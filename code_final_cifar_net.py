from torch import nn
import torch
from torch.nn.functional import gumbel_softmax
# from quantization import quantization, dequantization
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from code_final_modules import Encoder, Decoder_Recon, Decoder_Class, awgn, normalize, ResidualBlock
#from utils import count_percentage




def modulation(logits, device):
    eps = 1e-10
    num_cate = 8
    prob_z = gumbel_softmax(logits, hard=False)
    discrete_code = gumbel_softmax(logits, hard=True, tau=1.5)
    const = [-7, -5, -3, -1, 1, 3, 5, 7]
    const = torch.tensor(const).to(device)
    temp = discrete_code * const
    output = torch.sum(temp, dim=2)
    return output, prob_z


class Our_Net(nn.Module):
    def __init__(self, config):
        super(Our_Net, self).__init__()
        self.config = config

        self.num_category = 8
        self.encoder = Encoder(self.config)
        self.prob_convs = nn.Sequential(
            nn.Linear(config.trans_bit * 2 * 4 * 4, config.trans_bit * 2 * self.num_category),
            nn.ReLU(),
        )
        self.decoder_recon = Decoder_Recon(self.config)
        self.decoder_class = Decoder_Class(int(config.trans_bit * 2 / 2), int(config.trans_bit * 2 / 8))

        self.initialize_weights()

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=1)

    def reparameterize(self, probs, step):
        mod_method = self.config.mod_method
        code, prob_code = modulation(probs, self.device, step, mod_method)
        return code, prob_code

    # def forward(self, x, epoch):
    def forward(self, x):
        x_f = self.encoder(x).reshape(x.shape[0], -1)
        z = self.prob_convs(x_f).reshape(x.shape[0], -1, self.num_category)
        # code, prob_code = self.reparameterize(z, epoch)

        # power, z = normalize(code)

        # z_hat = awgn(0, z, self.device)
        # recon = self.decoder_recon(z_hat)
        # r_class = self.decoder_class(z_hat)
        recon = self.decoder_recon(z)

        # return code, prob_code, z, z_hat, r_class, recon
        return recon




