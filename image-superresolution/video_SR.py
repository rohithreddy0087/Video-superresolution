# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 11:31:21 2020

@author: Dell
"""

import cv2 
import os 

import argparse
import sys
import os

import torch
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from model import Generator
from PIL import Image
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
lr_imageSize = (120,120)
num_of_residual_blocks = 16
upfactor = 2

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

scale = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(lr_imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])

hr_img = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            ])

unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

generator = Generator(num_of_residual_blocks,upfactor)
generatorWeights = 'C:/Users/Dell/Desktop/SRGAN/generator_final.pth'
generator.load_state_dict(torch.load(generatorWeights))

def image_sr(high_res_real):
    low_res = torch.FloatTensor(1, 3, lr_imageSize[0], lr_imageSize[1])
    low_res[0] = scale(high_res_real)
    high_res_fake = generator(Variable(low_res))
    
    final_img_hr = unnormalize(high_res_fake.data[0])
    final_img_lr = unnormalize(low_res.data[0])
    save_image(final_img_lr,"C:/Users/Dell/Desktop/SRGAN/manual_testing/lr.jpg")
    save_image(final_img_hr,"C:/Users/Dell/Desktop/SRGAN/manual_testing/hr.jpg")
    return final_img_lr.numpy(),final_img_hr.numpy()
    
cam = cv2.VideoCapture("C:/Users/Dell/Desktop/SRGAN/video_input/Zoom -- how to lock meeting room (so you can keep using same ID).mp4") 
length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cam.get(cv2.CAP_PROP_FPS))
print("Number of Frames",length)
print("Height of Frame",height)
print("Width of Frame",width)
print("Number of frames per second",fps)

hr_video_width = 240
hr_video_height = 240
hr_video_fps = fps
video_name_hr = "output_hr.avi"
lr_video_width = 120
lr_video_height = 120
lr_video_fps = fps
video_name_lr = "output_lr.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_hr = cv2.VideoWriter(video_name_hr, fourcc,fps, (hr_video_width, hr_video_height))  
video_lr = cv2.VideoWriter(video_name_lr, fourcc,fps, (lr_video_width, lr_video_height))  
    
#high_img = np.zeros((240,240,3))
#low_img = np.zeros((120,120,3))
success,frame = cam.read() 
#plt.imshow(frame)
for i in range(1):
    if success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        low,high = image_sr(frame)
        low_img = low.transpose((1, 2, 0))
        high_img = high.transpose((1, 2, 0))
        print(frame.shape)
        print(high_img.shape)
        #plt.imshow(high_img)
        #low_img = cv2.cvtColor(low_img, cv2.COLOR_RGB2BGR)
        #high_img = cv2.cvtColor(high_img, cv2.COLOR_RGB2BGR)
        #plt.imshow(high_img.astype(np.uint8))
        video_hr.write(high_img)  
        video_lr.write(low_img)
        success, frame = cam.read()
        print(i)
cv2.destroyAllWindows()  
video_hr.release()
video_lr.release()