#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torch
from tqdm import tqdm
import pandas as pd
import os

from models import Generator, Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
# from utils import Logger
from utils import weights_init_normal
from datasets import RamImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--split', type=float, default=1, help='Train/Val split')
parser.add_argument('--output_dir', type=str, default="./output/", help='The output directory to save the model to')
parser.add_argument('--load_dir', type=str, default=None, help='If provided, loads the model from that directory')
parser.add_argument('--cycle_factor', type=float, default=10.0, help='The cycle factor to use')
parser.add_argument('--smooth', type=float, default=1, help='Smoothing factor')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

tensor_device = "cuda" if opt.cuda else "cpu"

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc).to(tensor_device)
netG_B2A = Generator(opt.output_nc, opt.input_nc).to(tensor_device)
netD_A = Discriminator(opt.input_nc).to(tensor_device)
netD_B = Discriminator(opt.output_nc).to(tensor_device)

# load variables
if opt.load_dir:
    print("Loading model")
    netG_A2B.load_state_dict(torch.load(os.path.join(opt.output_dir, 'netG_A2B.pth')))
    netG_B2A.load_state_dict(torch.load(os.path.join(opt.output_dir, 'netG_B2A.pth')))
    netD_A.load_state_dict(torch.load(os.path.join(opt.output_dir, 'netD_A.pth')))
    netD_B.load_state_dict(torch.load(os.path.join(opt.output_dir, 'netD_B.pth')))

# check if output dir exists
if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)


# init and set all models to train mode
netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

netG_A2B.train()
netG_B2A.train()
netD_A.train()
netD_B.train()

# Lossess
criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
smooth = opt.smooth
target_real = torch.full((opt.batchSize, 1), fill_value=smooth).float().to(tensor_device)
target_fake = torch.full_like(target_real, fill_value=1 - smooth).float().to(tensor_device)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(opt.size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataset = RamImageDataset(opt.dataroot, transforms_=transforms_, unaligned=opt.split == 1) # if there exists a split, then we dont want to take random samples from the dataloder
train_size = int(len(dataset) * opt.split)
train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
dataloader = DataLoader(train_set, 
                        batch_size=opt.batchSize, shuffle=False)#, num_workers=opt.n_cpu)

# Loss plot
# logger = Logger(opt.n_epochs, len(dataloader))
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    print("Epoch", epoch)
    epoch_hist = pd.DataFrame(columns=['epoch', 'loss_G', 'loss_G_identity', 'loss_G_GAN',
                        'loss_G_cycle', 'loss_D'])
    for batch in tqdm(dataloader):
        # Set model input
        real_A = batch['A'].to(tensor_device)
        real_B = batch['B'].to(tensor_device)

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*opt.cycle_factor

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*opt.cycle_factor

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        with torch.no_grad():
            batch_hist = {'epoch': torch.tensor([epoch]),'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A), 'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}
            batch_hist = {key: [round(value.detach().cpu().item(), 4)] for key, value in batch_hist.items()}
            epoch_hist = epoch_hist.append(pd.DataFrame(batch_hist))

    # after epoch, save hist to file
    output_filepath = os.path.join(opt.output_dir, "history.csv")
    epoch_hist = pd.DataFrame(epoch_hist.mean(axis=0)).transpose()
    epoch_hist.to_csv(output_filepath, index=False, mode="a", header=not os.path.exists(output_filepath))

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), os.path.join(opt.output_dir, 'netG_A2B.pth'))
    torch.save(netG_B2A.state_dict(), os.path.join(opt.output_dir, 'netG_B2A.pth'))
    torch.save(netD_A.state_dict(), os.path.join(opt.output_dir, 'netD_A.pth'))
    torch.save(netD_B.state_dict(), os.path.join(opt.output_dir, 'netD_B.pth'))
###################################
