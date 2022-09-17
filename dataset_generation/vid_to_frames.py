# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 22:40:04 2020

@author: Student
"""

import cv2
import os 
import shutil
f=" "
mypath = os.path.join(os.getcwd(),'dataset\\AB_mixed_testset\\videos')
vid = []

for f in os.listdir(mypath):
    vid.append(os.path.join(mypath,f))

folder=0
path = os.path.join(os.getcwd(),'dataset\\AB_mixed_testset')
file = open(os.path.join(path,"video.txt"), 'a')
for video in vid:
    folder= folder+1
    
    video_file=video
    vidcap = cv2.VideoCapture(video_file)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success,image = vidcap.read()
    count = 1
    if length > 148:
        try: 
            os.mkdir(os.path.join(path,"video"+str(folder)))
        except:
            shutil.rmtree(os.path.join(path,"video"+str(folder)))
            os.mkdir(os.path.join(path,"video"+str(folder)))
        os.chdir(os.path.join(path,"video"+str(folder)))  
        while success and count < 150:
          cv2.imwrite("frame"+'{0:03d}'.format(count)+".jpg", image)     # save frame as JPEG file      
          success,image = vidcap.read()
          file.write("video"+str(folder)+"/"+"frame"+'{0:03d}'.format(count)+".jpg")
          file.write("\n")
          count += 1
    print(folder)
