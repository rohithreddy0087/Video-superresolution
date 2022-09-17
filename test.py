# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:31:36 2020

@author: Student
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
from data import get_test_set
from autoEncoder import SRAutoencoder
import time

import cv2
import math
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
#parser.add_argument('--data_dir', type=str, default='D:/python/New/dataset/A_testset1')
parser.add_argument('--data_dir', type=str, default='D:/python/New/dataset/B_testset')
#parser.add_argument('--data_dir', type=str, default='D:/python/New/dataset/AB_mixed_testset')
parser.add_argument('--file_list', type=str, default='video.txt')
parser.add_argument('--other_dataset', type=bool, default=True, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=False, help="use future frame")
parser.add_argument('--nFrames', type=int, default=5)
parser.add_argument('--model_type', type=str, default='En')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--output', default='Results/4x_encoder/', help='Location to save checkpoint models')
#parser.add_argument('--model', default='weights/A_adam/2x_ENCODER_past_epoch_84.pth', help='sr pretrained base model')
#parser.add_argument('--model', default='weights/B_devibakthnagar/2x_ENCODER_B_epoch_34.pth', help='sr pretrained base model')
parser.add_argument('--model', default='weights/AB_mixed/2x_ENCODER_AB_mixed_epoch_34.pth', help='sr pretrained base model')


opt = parser.parse_args()
gpus_list = range(opt.gpus)


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)


print('===> Loading datasets')
test_set = get_test_set(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.file_list, opt.other_dataset, opt.future_frame)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model ', opt.model_type)
model = SRAutoencoder() 

model = torch.nn.DataParallel(model, device_ids=gpus_list)
model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained SR model is loaded.')


def eval():
    model.eval()
    count=1
    avg_psnr_predicted = 0.0
    for batch in testing_data_loader:
        stack =  batch[1].cuda(gpus_list[0])
        target = batch[2]

        t0 = time.time()
        prediction = model(stack)
        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
       
        #save_input_img(prediction.cpu().data, str(count),count, True)
        #save_input_img(input, str(count),count, False)
        # c = prediction.cpu().data
        # plt.figure()
        # plt.imshow(batch[0][0].squeeze().clamp(0, 1).numpy().transpose(1,2,0))
        # plt.title('input')
        # plt.figure()
        # plt.imshow(c.squeeze().clamp(0, 1).numpy().transpose(1,2,0))
        # plt.title('output')
        # print(s)
        
        prediction=prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)
        prediction = prediction*255.
        
        target = target.squeeze().numpy().astype(np.float32)
        target = target*255.
        
        
        psnr_predicted = PSNR(prediction,target, shave_border=opt.upscale_factor)
        avg_psnr_predicted += psnr_predicted
        if count % 149 == 0 :
          psnr_list.append(avg_psnr_predicted/149)
          avg_psnr_predicted = 0
        count+=1

    
    #print("PSNR_predicted=", avg_psnr_predicted/count)

def save_psnr(psnr,count,flag):
    # save img
    f = count % 149

    if f != 0 :
      name = 'video'+str(int(count/149)+1)
      img_name = str(f)
    else:
      name = 'video'+str(int(count/149))
      img_name = str(149)
    
def save_input_img(img, img_name,count,flag):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)

    # save img
    f = count % 149

    if f != 0 :
      name = 'video'+str(int(count/149)+1)
      img_name = str(f)
    else:
      name = 'video'+str(int(count/149))
      img_name = str(149)
    save_dir=os.path.join(opt.output+name+'_'+str(opt.upscale_factor)+'x')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.chdir(save_dir)
        os.mkdir('lr_input')
        os.mkdir('hr_output')
    
    if flag == 0: 
        save_fn = save_dir +'/lr_input/'+img_name+'.png'
    else:
        save_fn = save_dir +'/hr_output/'+img_name+'.png'
    
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    #pred = pred[1+shave_border:height - shave_border, 1+shave_border:width - shave_border, :]
    #gt = gt[1+shave_border:height - shave_border, 1+shave_border:width - shave_border, :]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

psnr_list = []
eval()