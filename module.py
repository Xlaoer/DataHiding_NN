import torch.nn as nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3
        )
        self.enc2 = nn.Conv2d(
            in_channels=8, out_channels=4, kernel_size=3
        )
    def forward(self, x):
       x = F.relu(self.enc1(x))
       x = F.relu(self.enc2(x))
       return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=4, out_channels=8, kernel_size=3
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=8, out_channels=3, kernel_size=3
        )
    def forward(self, x):
       x = F.relu(self.dec1(x))
       x = F.relu(self.dec2(x))
       return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.flatten = nn.Flatten()
    def forward(self, x):
        encoded_image = self.encoder(x)

        # Apply 64QAM modulation to the encoded representation
        flat_encoded = self.flatten(encoded_image)
        modulated_symbol = apply_64QAM_modulation(flat_encoded)

        # Reshape the modulated symbol and pass it through the decoder
        modulated_symbol_reshaped = modulated_symbol.view(-1, 4, 252, 252)
        decoded_image = self.decoder(modulated_symbol_reshaped)

        return decoded_image


def apply_64QAM_modulation(encoded_output):
    # Assuming encoded_output is a tensor with shape (batch_size, num_features)

    # Quantize the real and imaginary parts to 4 levels each
    quantized_real = torch.round(encoded_output[:, :32] / 0.5) * 0.5
    quantized_imag = torch.round(encoded_output[:, 32:] / 0.5) * 0.5

    # Map the quantized values to 64QAM symbols
    real_part = torch.index_select(quantized_real, 1, torch.arange(0, 32, 2))
    imag_part = torch.index_select(quantized_imag, 1, torch.arange(1, 32, 2))

    # Create complex numbers from real and imaginary parts
    complex_symbols = torch.complex(real_part, imag_part)

    # Map the complex symbols to 64QAM modulation
    modulation_table = torch.tensor([-3, -1, 1, 3], dtype=torch.float32)
    indices = torch.argmax(torch.abs(complex_symbols.unsqueeze(-1) - modulation_table), dim=-1)

    # Map the indices to 64QAM symbols
    modulated_symbols = modulation_table[indices]

    return modulated_symbols