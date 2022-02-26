
import os

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torch.nn.modules.activation import Tanh
import torch.nn.functional as F

from get_param import get_param
from get_dataset import get_cifar10
from generator import generator
from discriminator import discriminator
from utils import label2onehot, CrossEntropy_uniform, make_Loss_Graph

# Get Parameters
param = get_param()

epochs = param['epochs']
lr = param['lr']
batch_size = param['batch_size']
img_size = param['img_size']
channels = param['channels']
dataset_name = param['dataset_name']
category_num = param['category_num']
input_noise_num = param['input_noise_size']

# Prepare Datasets
_, data_loader = get_cifar10(batch_size, 2)

# Define Losses
b_loss = nn.BCELoss()
c_loss = nn.CrossEntropyLoss()
G_total_loss = []
D_total_loss = []

# Set Models
net_G = generator(input_noise_num, dataset_name)
net_D = discriminator(category_num, dataset_name)

# Set Optimizers
G_opt = optim.SGD(net_G.parameters(), lr=lr)
D_opt = optim.SGD(net_D.parameters(), lr=lr)


with tqdm(range(epochs)) as pbar_epochs:
  for epoch in pbar_epochs:
    pbar_epochs.set_description("[Epoch %d]" % (epoch+1))

    net_G.train()
    net_D.train()

    running_G_loss = 0.0
    running_D_loss = 0.0

    with tqdm(enumerate(data_loader), total=len(data_loader), leave=False) as pbar_loss:
      for i, (real_imgs, labels) in pbar_loss:
        num_b = real_imgs.size()[0]

        labels = label2onehot(labels)
        labels = labels.long()

        zs = torch.randn((num_b, input_noise_num))
        fake_imgs = net_G(zs)

        # Discriminatorの学習
        net_D.zero_grad()
        disc_b_realloss = b_loss(net_D(real_imgs)[0], torch.ones((num_b, 1)))
        disc_c_loss = c_loss(net_D(real_imgs)[1], labels)
        disc_b_fakeloss = b_loss(net_D(fake_imgs)[0], torch.zeros((num_b, 1)))
        D_loss = disc_b_realloss + disc_c_loss + disc_b_fakeloss
        D_loss.backward(retain_graph=True)
        D_opt.step()

        # Generatorの学習
        net_G.zero_grad()
        gen_b_loss = b_loss(net_D(fake_imgs)[0], torch.ones((num_b, 1)))
        gen_c_loss = CrossEntropy_uniform(net_D(fake_imgs)[1], num_b, category_num)
        G_loss = gen_b_loss + gen_c_loss
        G_loss.backward(retain_graph=True)
        G_opt.step()
    
    print('G_loss: {}, D_loss: {}'.format(
        G_loss/num_b,
        D_loss/num_b
        )
    )

    # loss graph用のリスト
    G_total_loss.append(G_loss.detach().numpy()/num_b)
    D_total_loss.append(D_loss.detach().numpy()/num_b)

    # 再構成可視化用
    os.mkdir('./visualize', exist_ok=True)
    save_image(real_imgs.detach()[:9], "./visualize/real_epoch{}.png".format(epoch+1), nrow=3, normalize=True)
    save_image(fake_imgs.detach()[:9], "./visualize/fake_epoch{}.png".format(epoch+1), nrow=3, normalize=True)

make_Loss_Graph(G_total_loss, D_total_loss)