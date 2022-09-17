# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 10:31:08 2020

@author: Dell
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread,imsave
from os import listdir
from os.path import isfile, join
import numpy as np
lr_path = 'C:/Users/Dell/Desktop/SRGAN/lr_images/'
hr_path = 'C:/Users/Dell/Desktop/SRGAN/hr_images/'
j = 1
train_data_length = 450
test_data_length = 50

train_x = np.empty([train_data_length,480,360,3])
train_y = np.empty([train_data_length,1280,720,3])
test_x = np.empty([test_data_length,480,360,3])
test_y = np.empty([test_data_length,1280,720,3])
for f in listdir(lr_path):
  if isfile(join(lr_path, f)):
    image_lr = imread(lr_path+"/"+str(j)+".jpg")
    image_hr = imread(hr_path+"/"+str(j)+".jpg")
    if j < train_data_length+1:
      train_x[j-1] = image_lr
      train_y[j-1] = image_hr
    elif j > train_data_length and j < 500 :
      test_x[j-train_data_length] = image_lr
      test_y[j-train_data_length] = image_hr
    else:
      break
    print(j)
    j = j + 1

print ("number of training examples = " + str(train_x.shape[0]))
print ("number of test examples = " + str(test_x.shape[0]))
print ("X_train shape: " + str(train_x.shape))
print ("Y_train shape: " + str(train_y.shape))
print ("X_test shape: " + str(test_x.shape))
print ("Y_test shape: " + str(test_y.shape))