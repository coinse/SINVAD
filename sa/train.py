import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import sys

from model import EasyClassifier, MnistClassifier

# Hyper-parameters
img_size = 28*28*1
h_dim = 800
num_epochs = 50
batch_size = 128
learning_rate = 2e-4

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)
test_dataset = torchvision.datasets.MNIST(root='../data',
                                     train=False,
                                     transform=transforms.ToTensor(),
                                     download=True)

# Data loader
train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size, 
                                                shuffle=True)
test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size, 
                                               shuffle=False)

if sys.argv[1] == 'conv':
    model = MnistClassifier(img_size = img_size)
else:
    model = EasyClassifier(img_size = img_size, h_dim = h_dim)
    
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i, (x, x_class) in enumerate(train_data_loader):
        # Forward pass
        if sys.argv[1] == 'conv':
            x = x.cuda()
        else:
            x = x.cuda().view(-1, img_size)
        class_logits = model(x)
        
        # Backprop and optimize
        loss = loss_fn(class_logits, x_class.cuda())
        
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
            if sys.argv[1] == 'conv':
                tx = tx.cuda()
            else:
                tx = tx.cuda().view(-1, img_size)
            tclass_logits = model(tx)
            _, mostprob_result = torch.max(tclass_logits, dim=1)
            total += tx.size(0)
            correct += torch.sum(mostprob_result == tx_class.cuda())
        print("%d/%d correct (%.2f %%)" % (correct, total, 100*float(correct)/total))

if sys.argv[1] == 'conv':
    torch.save(model.state_dict(), 'models/MNIST_conv_classifier.pth')
else:
    torch.save(model.state_dict(), 'models/MNIST_classifier.pth')
