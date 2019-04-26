#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:06:23 2018

@author: Soenke
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
 
imsize = (256, 64)
subsam_factor = 4
dim = 0

impulse = np.zeros(imsize[0])
impulse[imsize[0]//2] = 1

# 3d
#a_ft = np.zeros(imsize)
#a_ft[np.floor(imsize[0]/2-1), np.floor(imsize[1]/2), np.floor(imsize[2]/2)] = 1
#a_ft[np.floor(imsize[0]/2+1), np.floor(imsize[1]/2), np.floor(imsize[2]/2)] = 1
#a_ft[imsize[0]-2            , np.floor(imsize[1]/2), np.floor(imsize[2]/2)] = 1
#a_ft[2                      , np.floor(imsize[1]/2), np.floor(imsize[2]/2)] = 1

a_ft = np.zeros(imsize)
a_ft[imsize[0]//2-1, 0] = 1  # aliasing
a_ft[imsize[0]//2+1, 0] = 1  # aliasing
a_ft[imsize[0]-2, 0] = 1  # no aliasing
a_ft[2,           0] = 1  # no aliasing

a = np.fft.ifftn(a_ft)
if np.max(np.imag(a)) < 1e-5 * np.max(np.real(a)):
    a = np.real(a)
else:
    print('max imaginary part of a is ', np.max(np.imag(a)))

a_subsam1 = a[::subsam_factor,:]
#a_subsam2 = signal.decimate(a, 2, axis=0)

#num, den = signal.cheby1(4, 1, 0.2, 'low')
num, den = signal.cheby2(4, 25, 0.27, 'low')
#num, den = signal.butter(4, 0.16, 'low')
w, h = signal.freqz(num, den, imsize[0])
#w1, h1 = signal.freqz(num1, den1, imsize[0])
#a_filtered = signal.filtfilt(num, den, a, axis=0)
#a_subsam2 = a_filtered[::subsam_factor,:]
a_subsam2 = signal.decimate(a, subsam_factor, axis=0, ftype=signal.dlti(num, den))
#print(np.max(a_subsam2 - a_subsam3))  # 0.0
print(np.mean(a))
print(np.mean(a_subsam1))
print(np.mean(a_subsam2))
print()
print(np.max(a))
print(np.max(a_subsam1))
print(np.max(a_subsam2))
print()
print(np.min(a))
print(np.min(a_subsam1))
print(np.min(a_subsam2))
#print(np.max(a_subsam3))

fig, axs = plt.subplots(2,2)
axs[0,0].imshow(a_ft)
axs[0,1].imshow(a)
axs[1,0].imshow(np.abs(np.fft.fftn(a_subsam1)))
axs[1,1].imshow(a_subsam1)
fig, axs = plt.subplots(2,2)
axs[0,0].plot(w,np.abs(h))
#axs[0,0].plot(w1,np.abs(h1))
axs[0,0].axvline(w[imsize[0]//4], color='green') # cutoff frequency
axs[1,0].plot(signal.filtfilt(num, den, impulse))
#axs[1,0].plot(signal.filtfilt(num1, den1, impulse))
axs[0,1].imshow(np.abs(np.fft.fftn(a_subsam2)))
axs[1,1].imshow(a_subsam2)
plt.show()
