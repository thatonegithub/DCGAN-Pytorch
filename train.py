# run to train the network
# configure this file's parameters in 'options.py'

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from options import *
from utils import *
from generator import Generator
from discriminator import Discriminator

print("DCGAN warming up ...")


# Create the training data directory if it already does not exist
if not os.path.isdir(op_train_data_path):
    os.mkdir(op_train_data_path)

# Dataset & Dataloader
norm_values = [0.5] * op_nc

data_transforms = transform = transforms.Compose([transforms.Resize(op_image_size),
                                                  transforms.CenterCrop(
    op_image_size),
    transforms.ToTensor(),
    transforms.Normalize(norm_values, norm_values)])

# uses the MNIST dataset by default
dataset = torchvision.datasets.MNIST(
    root=op_train_data_path, train=True, transform=data_transforms, download=True)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=op_batch_size, shuffle=True, num_workers=op_train_workers)

device = get_device(op_gpu)

# Create the generator & the discriminator
netG = Generator(op_gpu, op_conv_channels, op_nz,
                 op_nc, op_enable_bias).to(device)
netD = Discriminator(op_gpu, op_conv_channels, op_nc,
                     op_enable_bias).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (op_gpu > 1):
    netG = nn.DataParallel(netG, list(range(op_gpu)))
    netD = nn.DataParallel(netD, list(range(op_gpu)))

# Optionally apply the weight-initialization mentioned in the paper,
netG.apply(dcgan_weights_init)
netD.apply(dcgan_weights_init)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(op_batch_size, op_nz, 1, 1, device=device)

# Setup label for real (1) and fake (0)
real_label = op_real_lables
fake_label = op_fake_lables

lr = op_train_learning_rate

# Create the optimizers. Optimizer beta values are extracted from the paper.
optimizerD = optim.Adam(netD.parameters(), lr=lr,
                        betas=(op_train_beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr,
                        betas=(op_train_beta1, 0.999))

# Load state to resume training

state_file_full_path = os.path.join(op_state_dir, op_state_file)
current_epoch = 0
current_batch = 0
loaded_batch_size = op_batch_size

if not os.path.isdir(op_state_dir):
    os.mkdir(op_state_dir)

if not op_train_load_state:
    if os.path.isfile(state_file_full_path):
        print('\033[93mwarning: load state option is set to (False) while a network state file already exists; further training will overwrite the file!!\033[0m')

if not os.path.isfile(state_file_full_path):
    op_train_load_state = False
    print('\033[93mno saved state detected. Training from scratch.\033[0m')

if op_train_load_state:

    print('Loading state from file: %s' % (state_file_full_path))

    state = torch.load(state_file_full_path, map_location=device)

    netG.load_state_dict(state['g_state_dict'])
    netD.load_state_dict(state['d_state_dict'])
    optimizerG.load_state_dict(state['og_state_dict'])
    optimizerD.load_state_dict(state['od_state_dict'])
    current_epoch = state['epoch']
    current_batch = state['batch']
    loaded_batch_size = state['batch_size']

    netG.train()
    netD.train()

if loaded_batch_size != op_batch_size:
    print('\033[93mwarning: batch size from the previous training differs from the current size. Recomputing batch number.\033[0m')


def save_state(epoch, batch):
    global netG, netD, optimizerD, optimizerG, state_file_full_path, op_batch_size

    torch.save({
        'g_state_dict': netG.state_dict(),
        'd_state_dict': netD.state_dict(),
        'og_state_dict': optimizerG.state_dict(),
        'od_state_dict': optimizerD.state_dict(),
        'epoch': epoch,
        'batch': batch,
        'batch_size': op_batch_size
    }, state_file_full_path)


iters = 0

data_iter = iter(data_loader)

start_batch = current_batch * (loaded_batch_size // op_batch_size)

for skip in range(start_batch):
    next(data_iter)

for epoch in range(current_epoch, op_train_num_epochs):

    # For each batch in the dataloader
    for i in range(start_batch, len(data_loader)):

        data = next(data_iter)

        netD.zero_grad()

        # batch of real data
        real_batch = data[0].to(device)

        # batch size
        b_size = real_batch.size(0)

        # real lables
        label = torch.full((b_size,), real_label,
                           dtype=torch.float, device=device)

        # test the discriminator with real images
        output = netD(real_batch).view(-1)

        # Calculate the loss of the discriminator on real images
        errD_real = criterion(output, label)

        # Calculate gradients for the discriminator
        errD_real.backward()

        # Normal noise that we will feed to the generator
        noise = torch.randn(b_size, op_nz, 1, 1, device=device)

        # Generate fake images from that noise
        fake = netG(noise)

        # change up the lables from real to fake
        label.fill_(fake_label)

        # Run the fake batch through the discriminator
        output = netD(fake.detach()).view(-1)

        # Calculate the loss of the discriminator on fake images
        errD_fake = criterion(output, label)

        # again compute the gradients for the discriminator this time for the fake pass
        errD_fake.backward()

        # total error of the discriminator just for logging
        errD = errD_real + errD_fake

        # update the model
        optimizerD.step()

        # Second Step

        netG.zero_grad()
        label.fill_(real_label)  # change the lables again to real

        # run the fake batch through the discriminator once more
        output = netD(fake).view(-1)

        # compute the loss of the generator. The more the discriminator classifies the fake images as real, the better.
        errG = criterion(output, label)

        # compute gradients for the generator
        errG.backward()

        # update the model
        optimizerG.step()

        if i % 1 == 0:
            print('\r[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f    '
                  % (epoch, op_train_num_epochs, i, len(data_loader),
                     errD.item(), errG.item()), end=' ')

        # save the models and the optimizers
        if iters % op_train_save_steps == 0 and iters > 0:
            save_state(epoch, i)

        iters += 1

    start_batch = 0
    data_iter = iter(data_loader)

if iters > 0:
    save_state(op_train_num_epochs, 0)

else:
    print('Already trained for %d epochs. Exiting...' % (op_train_num_epochs))
