import torch
import torch.nn as nn


def dcgan_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_device(gpu):
    return torch.device("cuda:0" if (
        torch.cuda.is_available() and gpu > 0) else "cpu")
