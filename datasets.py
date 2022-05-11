import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class RamImageDataset(Dataset):
        def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
            self.transform = transforms.Compose(transforms_)
            self.unaligned = unaligned

            self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
            self.tensors_A = []
            self.tensors_B = []
            for filename_A in self.files_A:
                im = Image.open(filename_A)
                # only append if right shape
                if np.array(im).shape == (256, 256, 3):
                    self.tensors_A.append(im)
                else:
                    print(f"WARNING: The image {filename_A} seems to be grayscale, thus skipping.")
            for filename_B in self.files_B:
                im = Image.open(filename_B)
                # only append if right shape
                if np.array(im).shape == (256, 256, 3):
                    self.tensors_B.append(im)
                else:
                    print(f"WARNING: The image {filename_B} seems to be grayscale, thus skipping.")

        def __getitem__(self, index):
            item_A = self.transform(self.tensors_A[index % len(self.tensors_A)])

            if self.unaligned:
                item_B = self.transform(self.tensors_B[random.randint(0, len(self.tensors_B) - 1)])
            else:
                item_B = self.transform(self.tensors_B[index % len(self.tensors_B)])

            return {'A': item_A, 'B': item_B}

        def __len__(self):
            return max(len(self.tensors_A), len(self.tensors_B))