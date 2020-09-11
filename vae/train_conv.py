import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from model import ConvVAE
from data_loader import get_GTSRB_loader

# create sample directory if not exists
sample_dir = 'samples_conv_svhn'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Hyper-parameters
img_size = (3, 32, 32)
h_dim = 4000
z_dim = 800
num_epochs = 500
batch_size = 128
learning_rate = 2e-4

# MNIST dataset
dataset = torchvision.datasets.SVHN(root='../data',
#                                      train=True,
                                     split='train',
                                     transform=transforms.ToTensor(),
                                     download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size, 
    shuffle=True
)

model = ConvVAE(img_size = img_size[1:], c_num = img_size[0], h_dim = h_dim, z_dim = z_dim)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (x, _) in enumerate(data_loader):
        # Forward pass
        x = x.cuda()
        x_reconst, mu, log_var = model(x)
        
        # Compute reconstruction loss and kl divergence
        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Backprop and optimize
        loss = reconst_loss + (epoch/num_epochs)*kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item()))
        
    with torch.no_grad():
        # Save the sampled images
        z = torch.randn(batch_size, z_dim).cuda()
        out = model.decode(z)
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))

        # Save the reconstructed images
        out, _, _ = model(x)
        x_concat = torch.cat([x, out], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))

    if ((epoch+1)%10) == 0:
        torch.save(model.state_dict(), 'models/SVHN_ConvEnD.pth')

