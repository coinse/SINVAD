#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from tqdm import trange

from sa.model import MnistClassifier
from vae.model import VAE

img_size = 28*28*1
torch.no_grad() # since nothing is trained here


### Prep (e.g. Load Models) ###
vae = VAE(img_size = 28*28, h_dim = 1600, z_dim = 400)
vae.load_state_dict(torch.load('./vae/models/MNIST_EnD.pth'))
vae.cuda()

classifier = MnistClassifier(img_size = img_size)
classifier.load_state_dict(torch.load('./sa/models/MNIST_conv_classifier.pth'))
classifier.eval()
classifier.cuda()
print("models loaded...")

test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor(), download=False)
test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
print("Data loader ready...")

### GA Params ###
gen_num = 500
pop_size = 50
best_left = 20
mut_size = 0.1
imgs_to_samp = 10000

all_img_lst = []
### multi-image sample loop ###
for img_idx in trange(imgs_to_samp):
    ### Sample image ###
    for i, (x, x_class) in enumerate(test_data_loader):
        samp_img = x[0:1]
        samp_class = x_class[0].item()

    img_enc, _ = vae.encode(samp_img.view(-1, img_size).cuda())

    ### Initialize optimization ###
    init_pop = [img_enc + 0.7 * torch.randn(1, 400).cuda() for _ in range(pop_size)]
    now_pop = init_pop
    prev_best = 999
    binom_sampler = torch.distributions.binomial.Binomial(probs=0.5*torch.ones(img_enc.size()))

    ### gogo GA !!! ###
    for g_idx in range(gen_num):
        indivs = torch.cat(now_pop, dim=0)
        dec_imgs = vae.decode(indivs).view(-1, 1, 28, 28)
        all_logits = classifier(dec_imgs)
        
        indv_score = [999 if all_logits[(i_idx, samp_class)] == max(all_logits[i_idx]) 
                      else torch.sum(torch.abs(indivs[i_idx] - img_enc))
                      for i_idx in range(pop_size)]

        best_idxs = sorted(range(len(indv_score)), key=lambda i: indv_score[i], reverse=True)[-best_left:]
        now_best = min(indv_score)
        if now_best == prev_best:
            mut_size *= 0.7
        else:
            mut_size = 0.1
        parent_pop = [now_pop[idx] for idx in best_idxs]
        
        k_pop = []
        for k_idx in range(pop_size-best_left):
            mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
            spl_idx = np.random.choice(400, size=1)[0]
            k_gene = torch.cat([parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:]], dim=1) # crossover

            # mutation
            diffs = (k_gene != img_enc).float()
            k_gene += mut_size * torch.randn(k_gene.size()).cuda() * diffs # random adding noise only to diff places
            # random matching to img_enc
            interp_mask = binom_sampler.sample().cuda()
            k_gene = interp_mask * img_enc + (1 - interp_mask) * k_gene

            k_pop.append(k_gene)
        now_pop = parent_pop + k_pop
        prev_best = now_best
        if mut_size < 1e-3:
            break # that's enough and optim is slower than I expected
        

    mod_best = parent_pop[-1].clone()
    final_bound_img = vae.decode(parent_pop[-1])
    final_bound_img = final_bound_img.detach().cpu().numpy()
    all_img_lst.append(final_bound_img)

all_imgs = np.vstack(all_img_lst)
np.save('bound_imgs_MNIST.npy', all_imgs)