# this file generates a video to demonstrate latent space interpolation. Run this after training the network.
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
from utils import get_device

seconds = op_ms_video_length
fps = op_ms_video_fps

if not os.path.isdir(op_ms_output_dir):
    os.mkdir(op_ms_output_dir)

device = get_device(op_gpu) if op_ms_video_gpu else 'cpu'

netG = Generator(0, op_conv_channels, op_nz, op_nc, op_enable_bias).to(device)

state = torch.load(os.path.join(op_state_dir, op_state_file), map_location=device)

netG.load_state_dict(state['g_state_dict'])
netG.eval()

noise = torch.randn(op_batch_size, op_nz, 1, 1, device=device)
target_noise = torch.randn(op_batch_size, op_nz, 1, 1, device=device)
noise_diff = target_noise - noise
steps = op_ms_video_steps

result = netG(noise).detach().cpu()

fig = plt.figure(figsize=(8, 8))
plt.title(op_ms_video_title)
plt.axis('off')
im = plt.imshow(np.transpose(vutils.make_grid(result, padding=2,
                pad_value=1.0, normalize=True), (1, 2, 0)), interpolation='none')

frame_id = 0

def animate_func(i):
    global noise
    global steps
    global frame_id
    global noise_diff
    global target_noise

    with torch.no_grad():
        batch = netG(noise).detach().cpu()
    image = vutils.make_grid(
        batch, padding=2, pad_value=1.0, normalize=True)

    im.set_array(np.transpose(image, (1, 2, 0)))

    if(frame_id % steps == 0):
        target_noise = torch.randn(op_batch_size, op_nz, 1, 1, device=device)
        noise_diff = target_noise - noise

    noise += noise_diff / steps

    print('\rFrame synthesized %d/%d     ' % (frame_id, seconds * fps), end=' ')
    frame_id += 1

    return [im]


anim = animation.FuncAnimation(
    fig, animate_func, frames=seconds*fps, interval=1000 / fps)

anim.save(os.path.join(op_ms_output_dir, op_ms_video_name),
          fps=fps, extra_args=['-vcodec', 'libx264'])