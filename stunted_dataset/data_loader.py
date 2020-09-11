#!/usr/bin/python

'''Loads data as numpy data form'''

import torch
import torchvision.datasets as dss
from torchvision import transforms
from torch.utils import data
import os
import numpy as np
from PIL import Image

class StuntedMNIST(dss.MNIST): # should change class... when dataset changes
    """Custom Wrapper to constrict dataset size for MNIST-like datasets"""
    def __init__(self, **kwargs):
        torch.manual_seed(1221)
        stunt1, stunt2 = kwargs['stunt_pair']
        mix_ratio = kwargs['stunt_ratio']
        del kwargs['stunt_pair'], kwargs['stunt_ratio']
        super(StuntedMNIST, self).__init__(**kwargs)
        stunt1_mask = (self.targets == stunt1)
        stunt2_mask = (self.targets == stunt2)
        meta_mask_maker = torch.distributions.bernoulli.Bernoulli(mix_ratio*torch.ones(stunt1_mask.size()))
        mix_mask = meta_mask_maker.sample().byte()
        mix_stunt1 = (mix_mask & stunt2_mask) | ((~mix_mask) & stunt1_mask)
        mix_stunt2 = (mix_mask & stunt1_mask) | ((~mix_mask) & stunt2_mask)
        
        
        self.targets[mix_stunt1] = stunt1
        self.targets[mix_stunt2] = stunt2
        x_unique = self.targets.unique(sorted=True)
        
        print('Stunted dataset details:')
        total_stunt1 = torch.sum(stunt1_mask.long())
        total_stunt2 = torch.sum(stunt2_mask.long())
        stunt1to2 = torch.sum(((mix_stunt1 != stunt1_mask) & (stunt1_mask == 1)).long())
        stunt2to1 = torch.sum(((mix_stunt2 != stunt2_mask) & (stunt2_mask == 1)).long())
        print(f'{stunt1}->{stunt2}: {stunt1to2}')
        print(f'{stunt2}->{stunt1}: {stunt2to1}')
        
    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__bases__[0].__name__, 'processed')
    
    def __len__(self):
        """Returns the total number of image files."""
        return len(self.data)
