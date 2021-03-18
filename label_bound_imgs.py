import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from sa.model import MnistClassifier
from vae.model import VAE

# MNIST settings
img_size = (1, 28, 28)
h_size = 1600
z_size = 400

classifier = MnistClassifier(img_size = img_size)
classifier.load_state_dict(torch.load('./sa/models/MNIST_conv_classifier.pth'))
classifier.eval()
classifier.cuda()
print('classifier loaded')

imgs = np.load('./data/bound_imgs_MNIST.npy')
imgs = imgs.reshape((-1, 1, 28, 28))

tensor_imgs = torch.Tensor(imgs).cuda()
labels = []
sep_num = 10

def label_func(arr):
    s_arr = arr
    return (s_arr[0] if s_arr[0] != 0 else sep_num)*sep_num + s_arr[1] #

for img_idx in tqdm(range(imgs.shape[0])):
    logits = classifier(tensor_imgs[img_idx:img_idx+1]).detach()
    bests = np.argsort(logits.cpu().numpy()[0])[-2:][::-1]
    labels.append(label_func(bests))

np_lbls = np.array(labels)
np.save('./data/bound_imgs_MNIST_labels.npy', np_lbls)
