# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:44:32 2020

@author: R Rohith Reddy
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread,imsave
from os import listdir
from os.path import isfile, join

path = 'C:/Users/Dell/Desktop/helen_1'
i = 1
for f in listdir(path):
  if isfile(join(path, f)):
    image = imread(path+"/"+f)
    image_resized_lr = resize(image, (960,720),anti_aliasing=True)
    if i < 451:
        imsave("C:/Users/Dell/Desktop/SRGAN/hr_images/images"+str(i)+".jpg",image_resized_lr)
    else:
        imsave("C:/Users/Dell/Desktop/SRGAN/hr_test/"+str(i)+".jpg",image_resized_lr)
    #image_resized_hr = resize(image, (1280,720),anti_aliasing=True)
    #imsave("C:/Users/Dell/Desktop/SRGAN/hr_images/"+str(i)+".jpg",image_resized_hr)
    print(i)
    i = i+1