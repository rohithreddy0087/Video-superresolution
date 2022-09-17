# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:03:31 2020

@author: Student
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:31:36 2020

@author: Student
"""

#from __future__ import print_function

import numpy as np
import os
import torch
from autoEncoder import SRAutoencoder
import time
from torchvision.transforms import Compose, ToTensor, Resize
import cv2
import math
from PIL import Image
import matplotlib.pyplot as plt
from download_test_video import test_on_youtube_video
scale=2
gpu_mode=False
seed=123
gpus=1
output ='Results/testing/'

## A_trained model path
#model_path = 'weights/2x_ENCODER_A_epoch_79.pth'
## B_trained model path
model_path = 'weights/2x_ENCODER_B_epoch_34.pth'
# AB_trained
model_path = 'weights/2x_ENCODER_AB_mixed_epoch_34.pth'


gpus_list = range(gpus)
nframe = 5

cuda = gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

link = 'https://www.youtube.com/watch?v=0mYKc6pYVM4'
test_on_youtube_video(link)

model = SRAutoencoder() 

model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
print('Pre-trained SR model is loaded.')


def eval(stack,count):
    stack = stack.unsqueeze(0)
    if cuda:
        stack =  stack.cuda(gpus_list[0])
    t0 = time.time()
    prediction = model(stack)
    t1 = time.time()
    
    if cuda:
        pred = prediction.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
    
    pred = prediction.data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)

    print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
    
    #prediction=prediction.cpu()
    # prediction = prediction.data[0].numpy().astype(np.float32)
    # prediction = prediction*255.
    
    # target = target.squeeze().numpy().astype(np.float32)
    # target = target*255.
    
    
    # psnr = PSNR(prediction,target)
    return pred        

    
def save_img(img, img_name,count,flag):
    save_dir=output
    if flag == 0: 
        save_fn = save_dir +'/lr/'+img_name+'.png'
        cv2.imwrite(save_fn,img)
    else:
        save_fn = save_dir +'/hr/'+img_name+'.png'
        cv2.imwrite(save_fn,img*255)

def PSNR(pred, gt):
    height, width = pred.shape[:2]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def push(lst,img):
    dup = [0]*len(lst)
    for i in range(len(lst)-1):
        dup[i+1] = lst[i]
    dup[0] = img
    return dup

def transform():
    return Compose([
        ToTensor(),
    ])

transform = transform()
video_file = 'C:/Users/Dell/Desktop/Presentation/Results/testing/input.mp4'
vidcap = cv2.VideoCapture(video_file)
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
fps    = int(vidcap.get(cv2.CAP_PROP_FPS))

size = (852, 480) 
result = cv2.VideoWriter('Results/testing/output.avi',0,fps,size) 

success,image = vidcap.read()
count = 1
model.eval()
while success:
    #input_orginal = 
    input_org = Image.fromarray(np.uint8(image)).convert('RGB')
    input_scaled = input_org.resize((426*2,240*2), Image.NEAREST)
    if count == 1:
        stack = [input_scaled]*nframe
    else:
        stack = push(stack,input_scaled)
    neigbor = [transform(j) for j in stack]
    st = torch.cat(neigbor,dim=0)
    prediction = eval(st,count)
    save_img(prediction, str(count),count, True)
    v = (prediction*255).astype(np.uint8)
    result.write(v)
    #save_img(image, str(count),count, False)
    success,image = vidcap.read()
    count = count + 1
result.release()