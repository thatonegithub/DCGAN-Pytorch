import torch
import torch.nn as nn

# unlike the paper, this model does not incorporate batch normalization
# add batch norm if needed

# the generator model


class Generator(nn.Module):
    def __init__(self, gpu, channels, noise_size, out_dim, have_bias=True):
        super(Generator, self).__init__()
        self.gpu = gpu
        self.main = nn.Sequential(

            nn.ConvTranspose2d(noise_size, channels, 4,
                               1, 0, bias=have_bias),  # 4
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(channels//1, channels//2,
                               4, 2, 1, bias=have_bias),  # 8
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(channels//2, channels//4, 4,
                               2, 1, bias=have_bias),  # 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(channels//4, channels//8, 4,
                               2, 1, bias=have_bias),  # 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(channels//8, out_dim,
                               4, 2, 1, bias=have_bias),  # 64

            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
