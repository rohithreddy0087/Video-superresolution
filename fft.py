# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 22:32:33 2020

@author: Dell
"""

import numpy as np
import cv2 
from scipy import signal,misc,interpolate
from matplotlib import pyplot as plt
import scipy

cap= cv2.VideoCapture("C:/Users/Dell/Desktop/BTP/RBPN/rbpn/dataset/test_set_3/videos/003.mp4")
i=0
frames=[]
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    frames.append(cv2.cvtColor((frame), cv2.COLOR_BGR2GRAY).astype("int"))
    i+=1

img_test = frames[0]
trans_3d = np.fft.fftn(img_test)

#t=np.log(np.abs(np.fft.fftshift(trans_3d))**2)
mag = np.real(trans_3d)  #np.abs(trans_3d)
phase = np.imag(trans_3d) #np.angle(trans_3d)

W, H = img_test.shape[:2]
new_W, new_H = (int(W/2),int(H/2))
xrange = lambda x: np.linspace(0, 1, x)

f = interpolate.interp2d(xrange(W), xrange(H), mag.T, kind="linear")
down_sample_mag = f(xrange(new_W), xrange(new_H))

f = interpolate.interp2d(xrange(W), xrange(H), phase.T, kind="linear")
down_sample_ph = f(xrange(new_W), xrange(new_H))

# =============================================================================
# down_sample_mag = scipy.ndimage.interpolation.zoom(mag,2)
# down_sample_ph = scipy.ndimage.interpolation.zoom(phase,2)
# =============================================================================

#down_sample = down_sample_mag * np.exp( 1j * down_sample_ph)
down_sample = down_sample_mag.T +  1j * down_sample_ph.T
#down_sample = trans_3d
out_img = np.fft.ifftn(down_sample)
f = interpolate.interp2d(xrange(W), xrange(H), img_test.T, kind="linear")
down_sample_ph = f(xrange(new_W), xrange(new_H))
out_img = down_sample_ph.T

plt.figure("Frame1")
plt.subplot(121)
plt.imshow(img_test,cmap=plt.get_cmap("gray"))
plt.subplot(122)
plt.imshow(out_img,cmap=plt.get_cmap("gray"))


cap.release()
cv2.destroyAllWindows()