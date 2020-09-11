import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from model import CifarClassifier

# Hyper-parameters
img_size = 32
h_dim = 800
num_epochs = 50
batch_size = 128
learning_rate = 2e-4
darc1_lambda = 1e-3

# MNIST dataset
train_dataset = torchvision.datasets.CIFAR10(root='../data',
                                     train=True,
#                                      split='train',
                                     transform=transforms.ToTensor(),
                                     download=True)
test_dataset = torchvision.datasets.CIFAR10(root='../data',
                                     train=False,
#                                      split='test',
                                     transform=transforms.ToTensor(),
                                     download=True)

# Data loader
train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size, 
                                                shuffle=True)
test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size, 
                                               shuffle=False)

model = CifarClassifier(img_size = img_size)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i, (x, x_class) in enumerate(train_data_loader):
        # Forward pass
        x = x.cuda()
        class_logits = model(x)
        
        # Backprop and optimize
        class_loss = loss_fn(class_logits, x_class.cuda())
        darc1_loss = torch.max(torch.sum(torch.abs(class_logits), dim=0))
        darc1_loss = (darc1_lambda / x.size(0)) * darc1_loss
        loss = class_loss + darc1_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], CE Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, len(train_data_loader), loss.item()))
        
    with torch.no_grad():
        total = 0.
        correct = 0.
        model.eval()
        for tx, tx_class in test_data_loader:
            tx = tx.cuda()
            tclass_logits = model(tx)
            _, mostprob_result = torch.max(tclass_logits, dim=1)
            total += tx.size(0)
            correct += torch.sum(mostprob_result == tx_class.cuda())
        print("%d/%d correct (%.2f %%)" % (correct, total, 100.*correct/total))
        model.train()

torch.save(model.state_dict(), 'models/CIFAR10_classifier.pth')