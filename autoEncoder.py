# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 20:31:07 2020

@author: Dell
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

class SRAutoencoder(nn.Module):
    def __init__(self,num_frame=5):
        super(SRAutoencoder, self).__init__()
       
        self.pool = nn.MaxPool2d(2, 2)
        #Encoder
        self.conv1 = nn.Conv2d(in_channels = 3*num_frame, out_channels = 128, kernel_size=3, stride=1,padding=1)  
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=3, stride=1,padding=1)  
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=3, stride=1,padding=1) 
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=3, stride=1,padding=1) 
        self.encoded = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=3, stride=1,padding=1) 
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')
        

        #Decoder
        self.t_conv1 = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.t_conv2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.t_conv3 = nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.t_conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.decoded = nn.Conv2d(in_channels = 128, out_channels = 3, kernel_size=3, stride=1,padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = self.pool(x2)
        x4 = F.relu(self.conv3(x3))
        x5 = F.relu(self.conv4(x4))
        x6 = self.pool(x5)
        x_en = F.relu(self.encoded(x6))
        x7 = self.upsample(x_en)
        x8 = F.relu(self.t_conv1(x7))
        x9 = F.relu(self.t_conv2(x8))
        x10 = x5+x9
        x11 = self.upsample(x10)
        x12 = F.relu(self.t_conv3(x11))
        x13 = F.relu(self.t_conv4(x12))
        x14 = x2+x13
        x15 = F.relu(self.decoded(x14))
        # print(x.size())
        # print(x1.size())
        # print(x2.size())
        # print(x3.size())
        # print(x4.size())
        # print(x5.size())
        # print(x6.size())
        # print(x_en.size())
        # print(x7.size())
        # print(x8.size())
        # print(x9.size())
        # print(x10.size())
        # print(x11.size())
        # print(x12.size())
        # print(x13.size())
        # print(x14.size())
        # print(x15.size())
        return x15

# class SRAutoencoder(nn.Module):
#     def __init__(self,num_frame=5):
#         super(SRAutoencoder, self).__init__()
       
#         #Encoder
#         self.conv1 = nn.Conv2d(in_channels = 3*num_frame, out_channels = 32, kernel_size=3, stride=1,padding=1)  
#         self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, stride=1,padding=1)  
#         self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, stride=1,padding=1) 
#         self.encoded = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=3, stride=1,padding=1) 
#         self.pool = nn.MaxPool2d(2, 2)

#         #Decoder
#         self.t_conv1 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2, padding = 0)
#         self.t_conv2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2, padding = 0)
#         self.t_conv3 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 2, stride = 2, padding = 0)
#         self.decoded = nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size = 2, stride = 2, padding = 0)


#     def forward(self, x):
#         b, c, h, w = x.size()
        
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)
#         x = F.relu(self.encoded(x))
#         x = self.pool(x)
#         x = F.relu(self.t_conv1(x))
#         x = F.relu(self.t_conv2(x))
#         x = F.relu(self.t_conv3(x))
#         x = F.sigmoid(self.decoded(x))
              
#         return x

# model = SRAutoencoder()
# print(model)
# v = torch.empty((32,15,64,64))
# c = model(v)