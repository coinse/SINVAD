import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import sys

from model import VGGNet, MobileNet, CifarClassifier

# Hyper-parameters
img_size = 28*28*1
h_dim = 800
num_epochs = 50
batch_size = 128
learning_rate = 2e-4

# MNIST dataset
train_dataset = torchvision.datasets.SVHN(root='../data',
                                     split='train',
                                     transform=transforms.ToTensor(),
                                     download=True)
test_dataset = torchvision.datasets.SVHN(root='../data',
                                     split='test',
                                     transform=transforms.ToTensor(),
                                     download=True)

# Data loader
train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size, 
                                                shuffle=True)
test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size, 
                                               shuffle=False)

if sys.argv[1] == 'vgg':
    model = VGGNet()
elif sys.argv[1] == 'mobile':
    model = MobileNet()
elif sys.argv[1] == 'custom':
    model = CifarClassifier()
else:
    raise ValueError(f'Unknown network type {sys.argv[1]}')
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i, (x, x_class) in enumerate(train_data_loader):
        # Forward pass
        x = x.cuda()#.view(-1, img_size)
        class_logits = model(x)
        
        # Backprop and optimize
        loss = loss_fn(class_logits, x_class.cuda())
        
        # darc1 regularizer (optional)
        darc1_loss = 0#1e-3*torch.max(torch.sum(torch.abs(class_logits), dim=0))
        loss = darc1_loss + loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], CE Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, len(train_data_loader), loss.item()))
        
    with torch.no_grad():
        total = 0.
        correct = 0.
        for tx, tx_class in test_data_loader:
            tx = tx.cuda()#.view(-1, img_size)
            tclass_logits = model(tx)
            _, mostprob_result = torch.max(tclass_logits, dim=1)
            total += tx.size(0)
            correct += torch.sum(mostprob_result == tx_class.cuda())
        print("%d/%d correct (%.2f %%)" % (correct, total, 100*float(correct)/total))

torch.save(model.state_dict(), f'models/SVHN_{sys.argv[1]}net.pth')
