# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:43:51 2020

@author: Dell
"""

import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tensorboard_logger import configure, log_value

from model import Generator, Discriminator, FeatureExtractor
from skimage.io import imread,imsave

datapath = 'C:/Users/Dell/Desktop/SRGAN/hr_images'
outpath = 'C:/Users/Dell/Desktop/SRGAN/trained_models'
workers = 0
batchSize = 8
lr_imageSize = (480,360)
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
dataset = datasets.ImageFolder(root= datapath, transform=hr_img)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,shuffle=True, num_workers=workers)
num_of_residual_blocks = 16
generator = Generator(num_of_residual_blocks,upfactor)
discriminator = Discriminator()
generatorWeights = 'C:/Users/Dell/Desktop/SRGAN/generator_final.pth'
discriminatorWeights = 'C:/Users/Dell/Desktop/SRGAN/discriminator_final.pth'
#generator.load_state_dict(torch.load(generatorWeights))
#discriminator.load_state_dict(torch.load(discriminatorWeights))

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

#configure('logs/' + dataset + '-' + str(batchSize) + '-' + str(generatorLR) + '-' + str(discriminatorLR), flush_secs=5)
#print('logs/' + dataset + '-' + str(batchSize) + '-' + str(generatorLR) + '-' + str(discriminatorLR))

low_res = torch.FloatTensor(batchSize, 3, lr_imageSize[0], lr_imageSize[1])

# Pre-train generator using raw MSE loss
print('Generator pre-training')
for epoch in range(2):
    mean_generator_content_loss = 0.0
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

        ######### Train generator #########
        generator.zero_grad()

        generator_content_loss = content_criterion(high_res_fake, high_res_real)
        mean_generator_content_loss += generator_content_loss.data[0]

        generator_content_loss.backward()
        optim_generator.step()

        ######### Status and display #########
        sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f' % (epoch, 2, i, len(dataloader), generator_content_loss.data[0]))

    sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f\n' % (epoch, 2, i, len(dataloader), mean_generator_content_loss/len(dataloader)))
    log_value('generator_mse_loss', mean_generator_content_loss/len(dataloader), epoch)

# Do checkpointing
torch.save(generator.state_dict(), '%s/generator_pretrain.pth' % outpath)
print("SRGAN Training")
# SRGAN training
optim_generator = optim.Adam(generator.parameters(), lr=generatorLR*0.1)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=discriminatorLR*0.1)

for epoch in range(nEpochs):
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
            target_real = Variable(torch.rand(batchSize,1)*0.5 + 0.7).cuda()
            target_fake = Variable(torch.rand(batchSize,1)*0.3).cuda()
        else:
            high_res_real = Variable(high_res_real)
            high_res_fake = generator(Variable(low_res))
            target_real = Variable(torch.rand(batchSize,1)*0.5 + 0.7)
            target_fake = Variable(torch.rand(batchSize,1)*0.3)
        
        ######### Train discriminator #########
        discriminator.zero_grad()

        discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                             adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
        mean_discriminator_loss += discriminator_loss.data[0]
        
        discriminator_loss.backward()
        optim_discriminator.step()

        ######### Train generator #########
        generator.zero_grad()

        real_features = Variable(feature_extractor(high_res_real).data)
        fake_features = feature_extractor(high_res_fake)

        generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
        mean_generator_content_loss += generator_content_loss.data[0]
        generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), ones_const)
        mean_generator_adversarial_loss += generator_adversarial_loss.data[0]

        generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss
        mean_generator_total_loss += generator_total_loss.data[0]
        
        generator_total_loss.backward()
        optim_generator.step()   
        
        ######### Status and display #########
        sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (epoch,nEpochs, i, len(dataloader),
        discriminator_loss.data[0], generator_content_loss.data[0], generator_adversarial_loss.data[0], generator_total_loss.data[0]))
    
    sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (epoch, nEpochs, i, len(dataloader),
    mean_discriminator_loss/len(dataloader), mean_generator_content_loss/len(dataloader), 
    mean_generator_adversarial_loss/len(dataloader), mean_generator_total_loss/len(dataloader)))

    log_value('generator_content_loss', mean_generator_content_loss/len(dataloader), epoch)
    log_value('generator_adversarial_loss', mean_generator_adversarial_loss/len(dataloader), epoch)
    log_value('generator_total_loss', mean_generator_total_loss/len(dataloader), epoch)
    log_value('discriminator_loss', mean_discriminator_loss/len(dataloader), epoch)

    # Do checkpointing
    torch.save(generator.state_dict(), '%s/generator_final.pth' % outpath)
    torch.save(discriminator.state_dict(), '%s/discriminator_final.pth' % outpath)