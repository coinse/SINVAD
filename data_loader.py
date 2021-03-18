#!/usr/bin/python

'''Loads data as numpy data form'''

import torch
import torchvision.datasets as dss
from torchvision import transforms
from torch.utils import data
import os
import numpy as np
from PIL import Image

class LabeledBoundaryData(dss.MNIST): # should change class when dataset changes
    """Custom Wrapper to read boundary images under PyTorch framework"""
    def __init__(self, **kwargs):
        self.bound_data_file = kwargs['bound_data']
        del kwargs['bound_data']
        self.bound_label_file = kwargs['bound_label']
        del kwargs['bound_label']
        self.seed_provided = 'seed_data' in kwargs
        if self.seed_provided:
            self.seed_data_file = kwargs['seed_data']
            del kwargs['seed_data']
        self.transform = kwargs['transform']
#         super(LabeledBoundaryData, self).__init__(**kwargs)
        
        bound_imgs = np.load(self.bound_data_file)
        bound_labels = np.load(self.bound_label_file)
        bound_imgs = bound_imgs.reshape((-1, 28, 28))
        
        self.bound_data = torch.Tensor(255*bound_imgs).byte()
        self.bound_targets = torch.Tensor(bound_labels).long()
        self.data = self.bound_data
        self.targets = self.bound_targets
        
        if self.seed_provided:
            seed_imgs = np.load(self.seed_data_file)
            seed_imgs = seed_imgs.reshape((-1, 28, 28))
            self.seed_data = torch.Tensor(255*seed_imgs).byte()
            self.seeds = self.seed_data
        
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
        
        if self.seed_provided:
            seed_img = self.seeds[index]
            seed_img = Image.fromarray(seed_img.numpy(), mode='L')
            if self.transform is not None:
                seed_img = self.transform(seed_img)
            return img, seed_img, target
        else:
            return img, target
    
    def __len__(self):
        """Returns the total number of image files."""
        return len(self.data)
