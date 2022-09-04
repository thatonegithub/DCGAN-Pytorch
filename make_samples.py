# this file generates an image using the generator. Run this after training the network.
# configure this file's parameters in 'options.py'

import torch
import torch.nn as nn
import torchvision.utils as vutils

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from generator import Generator
from options import *

print('Making a sample image (%d images in one) at %s' % (op_batch_size, os.path.join(op_ms_output_dir, op_ms_output_name)))

if not os.path.isdir(op_ms_output_dir):
    os.mkdir(op_ms_output_dir)

device = op_ms_device

netG = Generator(0, op_conv_channels, op_nz, op_nc, op_enable_bias).to(device)

state = torch.load(os.path.join(op_state_dir, op_state_file), map_location=device)

netG.load_state_dict(state['g_state_dict'])
netG.eval()

noise = torch.randn(op_batch_size, op_nz, 1, 1, device=device)

result = netG(noise).detach().cpu()

plt.figure(figsize=(8, 8))
plt.title(op_ms_output_title)
plt.axis('off')
plt.imshow(np.transpose(vutils.make_grid(result, padding=2, pad_value=1.0, normalize=True),(1,2,0)), interpolation='none')

plt.savefig(os.path.join(op_ms_output_dir, op_ms_output_name))

plt.show()