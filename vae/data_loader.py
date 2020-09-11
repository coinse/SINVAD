#!/usr/bin/python

'''Loads data as numpy data form'''

import torch
import torchvision.datasets as dss
from torchvision import transforms
from torch.utils import data
import os
import numpy as np
from PIL import Image

class GTSRBFolder(data.Dataset):
    """GTSRB folder"""
    def __init__(self, root, transform=None, max_per_class=-1):
        """Initializes image paths and preprocessing module."""
        dir_names = list(map(lambda x: int(x), os.listdir(root)))
        self.dir_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_paths = []
        for d_idx, dir_path in enumerate(self.dir_paths):
            img_in_dir = list(map(lambda x: (os.path.join(dir_path, x), dir_names[d_idx]), os.listdir(dir_path)))
            self.image_paths += img_in_dir
        self.image_paths = list(filter(lambda x: x[0][-4:] == '.ppm', self.image_paths))
        self.transform = transform
        
    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path, image_label = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, image_label
    
    def __len__(self):
        """Returns the total number of image files."""
        return len(self.image_paths)

def get_GTSRB_loader(image_path, batch_size, num_workers = 2):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    dataset = GTSRBFolder(image_path, transform=transform)
    data_loader = data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers
    )
    return data_loader