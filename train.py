# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 20:30:50 2020

@author: Dell
"""

from __future__ import print_function
import argparse
from math import log10
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_eval_set
#from autoEncoder import ConvAutoencoder
from autoEncoder import SRAutoencoder
import time

import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

data_dir_path = "D:/python/New/dataset/AB_mixed_dataset"
file_list_path = "video.txt"
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=35, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
parser.add_argument('--lr', type=float, default=4e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default=data_dir_path)
parser.add_argument('--file_list', type=str, default=file_list_path)
parser.add_argument('--other_dataset', type=bool, default=True, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=False, help="use future frame")
parser.add_argument('--nFrames', type=int, default=5)
parser.add_argument('--patch_size', type=int, default=0, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=False)
parser.add_argument('--model_type', type=str, default='ENCODER_AB_mixed')
#parser.add_argument('--pretrained_sr', default=pretrained_path, help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='VIDEO', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
cudnn.benchmark = True
print(opt)

def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        stack =  batch[1].cuda(gpus_list[0])
        target = batch[2].cuda(gpus_list[0])

        # plt.figure()
        # plt.imshow(target[0].cpu().numpy().transpose(1,2,0))
        # plt.title('input')
        # plt.figure()
        # plt.imshow(batch[0][0].squeeze().clamp(0, 1).numpy().transpose(1,2,0))
        # plt.title('output')
        # print(s)
        t0 = time.time()
        pred = model(stack)
        t1 = time.time()
        optimizer.zero_grad()
        loss = criterion1(pred,target)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        loss_iter.append("===> Epoch[{}]({}/{}): Loss: {:.4f}|| Timer: {:.4f} sec.\n".format(epoch, iteration, len(training_data_loader), loss.item(), (t1 - t0)))
        print("===> Epoch[{}]({}/{}): Loss: {:.4f}|| Timer: {:.4f} sec.\n".format(epoch, iteration, len(training_data_loader), loss.item(), (t1 - t0)))
    loss_epoch.append("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    model_out_path = opt.save_folder+str(opt.upscale_factor)+'x_'+opt.model_type+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)


print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation, opt.file_list, opt.other_dataset, opt.patch_size, opt.future_frame)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

print('===> Building model ', opt.model_type)
model = SRAutoencoder() 

model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion1 = nn.MSELoss()


# print('---------- Networks architecture -------------')
# print_network(model)
# print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)

    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion1 = criterion1.cuda(gpus_list[0])
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
loss_iter = []
loss_epoch = []
for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    train(epoch)
    
    # if (epoch+1) % (opt.nEpochs/2) == 0:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] /= 10.0
    #     print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
            
    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)
    
