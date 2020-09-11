import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import sys

from model import MnistClassifier
from data_loader import StuntedMNIST

# Hyper-parameters
img_size = 28*28*1
h_dim = 800
num_epochs = 50
batch_size = 128
learning_rate = 2e-4
norm_train = False
stunt_ratio = float(sys.argv[1])
expr_name = f'stunt_{stunt_ratio:.3f}'
print(f'Experiment name: {expr_name}')

# MNIST dataset
train_dataset = StuntedMNIST(root='../data',
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True,
                             stunt_pair=(0, 1),
                             stunt_ratio=stunt_ratio)
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

model = MnistClassifier(img_size = img_size)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# helper function
def print_error_matrix(test_data, classifier):
    import matplotlib.pyplot as plt
    error_matrix = np.zeros((10, 10))
    for tx, tx_class in test_data_loader:
        tx = tx.cuda()
        tclass_logits = model(tx)
        _, mostprob_result = torch.max(tclass_logits, dim=1)
        mostprob_result = mostprob_result.cpu()
        for true_class, pred_class in zip(tx_class, mostprob_result):
            error_matrix[true_class.item(), pred_class.item()] += 1
    print(error_matrix)
    fig, ax = plt.subplots()
    ax.imshow(error_matrix, cmap='hot', interpolation='nearest')
    for x in range(10):
        for y in range(10):
            ax.text(x, y, str(int(error_matrix[y][x])), 
                     horizontalalignment='center', verticalalignment='center',
                     color='blue')
    ax.set_xlabel('Classified As...')
    ax.set_ylabel('Is...')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    plt.savefig(f'MNIST_{expr_name}.png', bbox_inches='tight')

for epoch in range(num_epochs):
    for i, (x, x_class) in enumerate(train_data_loader):
        # Forward pass
        x = x.cuda()#.view(-1, img_size)
        class_logits = model(x)
        if norm_train:
            class_logits = class_logits[:, :10]
        
        # Backprop and optimize
        class_loss = loss_fn(class_logits, x_class.cuda())
        darc1_loss = 1e-3*torch.max(torch.sum(torch.abs(class_logits), dim=0))
        loss = class_loss + 0*darc1_loss
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

print_error_matrix(test_data_loader, model)
torch.save(model.state_dict(), f'models/MNIST_{expr_name}.pth')
