#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import RamImageDataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--load_dir', type=str, default='./output/', help='Dir where to load the models from')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
tensor_device = "cuda" if opt.cuda else "cpu"

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc).to(tensor_device)
netG_B2A = Generator(opt.output_nc, opt.input_nc).to(tensor_device)

# Load state dicts
netG_A2B.load_state_dict(torch.load(os.path.join(opt.load_dir, "netG_A2B.pth")))
netG_B2A.load_state_dict(torch.load(os.path.join(opt.load_dir, "netG_B2A.pth")))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()


# Dataset loader
transforms_ = [ transforms.ToTensor() ,
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
inverse_transform = lambda image: (image +1) *0.5

dataset = RamImageDataset(opt.dataroot, transforms_=transforms_, unaligned=False, mode="test")
dataloader = DataLoader(dataset, 
                        batch_size=opt.batchSize, shuffle=False)
###################################

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists(os.path.join(opt.load_dir, "A")):
    os.makedirs(os.path.join(opt.load_dir, "A"))
if not os.path.exists(os.path.join(opt.load_dir, "B")):
    os.makedirs(os.path.join(opt.load_dir, "B"))

for i, batch in enumerate(tqdm(dataloader)):
    real_A = batch['A'].to(tensor_device)
    real_B = batch['B'].to(tensor_device)

    # Generate output
    fake_B = inverse_transform(netG_A2B(real_A))
    fake_A = inverse_transform(netG_B2A(real_B))

    # Save image files
    save_image(fake_A, os.path.join(opt.load_dir, f"A/fake_A_{i}.png"))
    save_image(fake_B, os.path.join(opt.load_dir, f"B/fake_B_{i}.png"))

###################################
