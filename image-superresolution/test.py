# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 23:10:38 2020

@author: Dell
"""

import argparse
import sys
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models import Generator, Discriminator, FeatureExtractor


datapath = '/content/drive/My Drive/SRGAN/hr_images'
outpath = '/content/drive/My Drive/SRGAN/trained_models'
workers = 0
batchSize = 10
lr_imageSize = (120,120)
upfactor = 2
nEpochs = 100

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(lr_imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])

hr_img = transforms.Compose([
                            transforms.ToTensor(),
                            ])

unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])
dataset = datasets.ImageFolder(root= datapath,transform=hr_img)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,shuffle=True, num_workers=workers)

num_of_residual_blocks = 16
generator = Generator(num_of_residual_blocks,upfactor)
discriminator = Discriminator()
generatorWeights = '/content/drive/My Drive/SRGAN/generator_final.pth'
discriminatorWeights = '/content/drive/My Drive/SRGAN/discriminator_final.pth'
generator.load_state_dict(torch.load(generatorWeights))
discriminator.load_state_dict(torch.load(discriminatorWeights))

# For the content loss
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

ones_const = Variable(torch.ones(batchSize, 1))


if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    feature_extractor.cuda()
    content_criterion.cuda()
    adversarial_criterion.cuda()
    ones_const = ones_const.cuda()

generatorLR = 0.0001
discriminatorLR = 0.0001
optim_generator = optim.Adam(generator.parameters(), lr=generatorLR)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=discriminatorLR)

configure('logs/' + datapath + '-' + str(batchSize) + '-' + str(generatorLR) + '-' + str(discriminatorLR), flush_secs=5)
#print('logs/' + dataset + '-' + str(batchSize) + '-' + str(generatorLR) + '-' + str(discriminatorLR))

low_res = torch.FloatTensor(batchSize, 3, lr_imageSize[0], lr_imageSize[1])

print("Started Testing")

mean_generator_content_loss = 0.0
mean_generator_adversarial_loss = 0.0
mean_generator_total_loss = 0.0
mean_discriminator_loss = 0.0

for i, data in enumerate(dataloader):
    # Generate data
    high_res_real, _ = data

    # Downsample images to low resolution
    for j in range(batchSize):
        low_res[j] = scale(high_res_real[j])
        high_res_real[j] = normalize(high_res_real[j])
    
    # Generate real and fake inputs
    if torch.cuda.is_available():
        high_res_real = Variable(high_res_real.cuda())
        high_res_fake = generator(Variable(low_res).cuda())
    else:
        high_res_real = Variable(high_res_real)
        high_res_fake = generator(Variable(low_res))


    discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                         adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
    mean_discriminator_loss += discriminator_loss.data
    

    ######### Train generator #########

    real_features = Variable(feature_extractor(high_res_real).data)
    fake_features = feature_extractor(high_res_fake)

    generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
    mean_generator_content_loss += generator_content_loss.data
    generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), ones_const)
    mean_generator_adversarial_loss += generator_adversarial_loss.data

    generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss
    mean_generator_total_loss += generator_total_loss.data
    

    ######### Status and display #########
    sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (epoch,nEpochs, i, len(dataloader),
    discriminator_loss.data, generator_content_loss.data, generator_adversarial_loss.data, generator_total_loss.data))
    
    for j in range(batchSize):
        save_image(unnormalize(high_res_real.data[j]), 'output/high_res_real/' + str(i*batchSize + j) + '.png')
        save_image(unnormalize(high_res_fake.data[j]), 'output/high_res_fake/' + str(i*batchSize + j) + '.png')
        save_image(unnormalize(low_res[j]), 'output/low_res/' + str(i*batchSize + j) + '.png')
    
sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (epoch, nEpochs, i, len(dataloader),
mean_discriminator_loss/len(dataloader), mean_generator_content_loss/len(dataloader), 
mean_generator_adversarial_loss/len(dataloader), mean_generator_total_loss/len(dataloader)))


