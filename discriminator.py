import torch
import torch.nn as nn

# the discriminator model

# unlike the paper, this model does not incorporate batch normalization
# add batch norm if needed


class Discriminator(nn.Module):
    def __init__(self, gpu, channels, in_dim, have_bias=True):
        super(Discriminator, self).__init__()
        self.gpu = gpu
        self.main = nn.Sequential(

            nn.Conv2d(in_dim, channels//8, 4, 2, 1, bias=have_bias),  # 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels//8, channels//4, 4,
                      2, 1, bias=have_bias),  # 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels//4, channels//2, 4,
                      2, 1, bias=have_bias),  # 8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels//2, channels//1, 4,
                      2, 1, bias=have_bias),  # 4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels//1, 1, 4, 1, 0, bias=have_bias),  # 1

            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
