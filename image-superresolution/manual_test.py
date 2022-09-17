# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 00:18:34 2020

@author: Dell
"""

import argparse
import sys
import os

import torch
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from model import Generator
from skimage.io import imread,imsave
from PIL import Image
from torchvision.utils import save_image

lr_imageSize = (120,120)
num_of_residual_blocks = 16
upfactor = 2

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

scale = transforms.Compose([
                            transforms.Resize(lr_imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])

hr_img = transforms.Compose([
                            transforms.ToTensor(),
                            ])

unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

generator = Generator(num_of_residual_blocks,upfactor)
generatorWeights = 'C:/Users/Dell/Desktop/SRGAN/generator_final.pth'
generator.load_state_dict(torch.load(generatorWeights))
low_res = torch.FloatTensor(1, 3, lr_imageSize[0], lr_imageSize[1])

path = 'C:/Users/Dell/Desktop/rrr3.jpg'
high_res_real = Image.open(path)
low_res[0] = scale(high_res_real)
high_res_fake = generator(Variable(low_res))

final_img_hr = unnormalize(high_res_fake.data[0])
final_img_lr = unnormalize(low_res.data[0])

save_image(final_img_lr,"C:/Users/Dell/Desktop/SRGAN/manual_testing/lr.jpg")
save_image(final_img_hr,"C:/Users/Dell/Desktop/SRGAN/manual_testing/hr.jpg")