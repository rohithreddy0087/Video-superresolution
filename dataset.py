import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
#import pyflow
from skimage import img_as_float
from random import randrange
import os.path

from torchvision.transforms import Compose, ToTensor, Resize


def transformT():
    return Compose([
        ToTensor(),
    ])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath, nFrames, scale, other_dataset):
    seq = [i for i in range(1,nFrames)]
    #random.shuffle(seq) #if random sequence
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'),scale)
        input=target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        input=input.resize((int(input.size[0]*scale),int(input.size[1]*scale)), Image.NEAREST)
        char_len = len(filepath)
        neigbor=[]
        target_neigbor = []

        for i in seq:
            index1 = int(filepath[char_len-7:char_len-4])-i
            file_name1=filepath[0:char_len-7]+'{0:03d}'.format(index1)+'.jpg'
            if os.path.exists(file_name1):
                temp_target = modcrop(Image.open(file_name1).convert('RGB'),scale)
                temp = modcrop(Image.open(file_name1).convert('RGB'),scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                temp =temp.resize((int(temp.size[0]*scale),int(temp.size[1]*scale)), Image.NEAREST)
                neigbor.append(temp)
                target_neigbor.append(temp_target)
            else:
                #print('neigbor frame is not exist')
                temp = input
                neigbor.append(temp)
    else:
        target = modcrop(Image.open(join(filepath,'im'+str(nFrames)+'.png')).convert('RGB'), scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = [modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC) for j in reversed(seq)]
        n_tar = [modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale) for j in reversed(seq)]
    return target, input, neigbor

def load_img_future(filepath, nFrames, scale, other_dataset):
    tt = int(nFrames/2)
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'),scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]
        if nFrames%2 == 0:
            seq = [x for x in range(-tt,tt) if x!=0] # or seq = [x for x in range(-tt+1,tt+1) if x!=0]
        else:
            seq = [x for x in range(-tt,tt+1) if x!=0]
        #random.shuffle(seq) #if random sequence
        for i in seq:
            index1 = int(filepath[char_len-7:char_len-4])+i
            file_name1=filepath[0:char_len-7]+'{0:03d}'.format(index1)+'.jpg'
            
            if os.path.exists(file_name1):
                temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame- is not exist')
                temp=input
                neigbor.append(temp)
            
    else:
        target = modcrop(Image.open(join(filepath,'im4.png')).convert('RGB'),scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = []
        seq = [x for x in range(4-tt,5+tt) if x!=4]
        #random.shuffle(seq) #if random sequence
        for j in seq:
            neigbor.append(modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC))
    return target, input, neigbor


def rescale_flow(x,max_range,min_range):
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range

def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih%modulo);
    iw = iw - (iw%modulo);
    img = img.crop((0, 0, ih, iw))
    img = img.resize((128,128),Image.BICUBIC)
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir,nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame, transform=None):
        super(DatasetFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.other_dataset = other_dataset
        self.patch_size = patch_size
        self.future_frame = future_frame

    def __getitem__(self, index):
        if self.future_frame:
            target, input, neigbor = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        else:
            target, input, neigbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        neigbor.insert(0,input)
        if self.transform:
            neigbor = [self.transform(j) for j in neigbor]
            input = self.transform(input)
            target = self.transform(target)
        st = torch.cat(neigbor,dim=0)   
        return input,st,target

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=None):
        super(DatasetFromFolderTest, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.other_dataset = other_dataset
        self.future_frame = future_frame

    def __getitem__(self, index):
        if self.future_frame:
            target, input, neigbor = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        else:
            target, input, neigbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        neigbor.insert(0,input)
        if self.transform:
            neigbor = [self.transform(j) for j in neigbor]
            input = self.transform(input)
            target = self.transform(target)
        st = torch.cat(neigbor,dim=0)   
        return input,st,target

    def __len__(self):
        return len(self.image_filenames)
