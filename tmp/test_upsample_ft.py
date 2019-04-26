#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 11:05:35 2018

@author: Soenke
"""
## Fourier upsampling

# The values in the result follow so-called “standard” order: 
# If A = fft(a, n), 

# then A[0] contains the zero-frequency term (the sum of the signal), which is 
# always purely real for real inputs. 

# Then A[1:n/2] contains the positive-frequency terms, 

# and A[n/2+1:] contains the negative-frequency terms, in order of decreasingly 
# negative frequency.

# For an even number of input points, A[n/2] represents both positive and 
# negative Nyquist frequency, and  is also purely real for real input. 
# For an odd number of input points, A[(n-1)/2] contains the largest positive 
# frequency, while A[(n+1)/2] contains the largest negative frequency. 
# The routine np.fft.fftfreq(n) returns an array giving the frequencies of 
# corresponding elements in the output. 
# The routine np.fft.fftshift(A) shifts transforms and their frequencies to put 
#the zero-frequency components in the middle, and np.fft.ifftshift(A) undoes 
# that shift.

from scipy import io
from os import path
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

# load data:
imdir = "/Users/Soenke/Documents/Uni/Master/4. Master/Masterarbeit/gits/code/testimages/3D/chromosome/"
np_im = io.loadmat(path.join(imdir, "im.mat"))["im"][0][0][0]
im = tf.constant(np_im)

upsam_factor = 2
sh = np.array(im.shape.as_list())
f_ny_old = sh//2

im_cmplx = tf.complex(im, tf.zeros(im.shape))
im_ft = tf.fft3d(im_cmplx)

# not transferable to tf
##im_ft_pad = tf.zeros(upsam_factor*sh, dtype=tf.complex64)
##im_ft_pad[:f_ny_old[0],  :f_ny_old[1],  :f_ny_old[2] ] = im_ft[:f_ny_old[0],  :f_ny_old[1],  :f_ny_old[2]]
##im_ft_pad[:f_ny_old[0],  :f_ny_old[1],  -f_ny_old[2]:] = im_ft[:f_ny_old[0],  :f_ny_old[1],  -f_ny_old[2]:]
##im_ft_pad[:f_ny_old[0],  -f_ny_old[1]:, :f_ny_old[2] ] = im_ft[:f_ny_old[0],  -f_ny_old[1]:, :f_ny_old[2]]
##im_ft_pad[:f_ny_old[0],  -f_ny_old[1]:, -f_ny_old[2]:] = im_ft[:f_ny_old[0],  -f_ny_old[1]:, -f_ny_old[2]:]
##im_ft_pad[-f_ny_old[0]:, :f_ny_old[1],  :f_ny_old[2] ] = im_ft[-f_ny_old[0]:, :f_ny_old[1],  :f_ny_old[2]]
##im_ft_pad[-f_ny_old[0]:, :f_ny_old[1],  -f_ny_old[2]:] = im_ft[-f_ny_old[0]:, :f_ny_old[1],  -f_ny_old[2]:]
##im_ft_pad[-f_ny_old[0]:, -f_ny_old[1]:, :f_ny_old[2] ] = im_ft[-f_ny_old[0]:, -f_ny_old[1]:, :f_ny_old[2]]
##im_ft_pad[-f_ny_old[0]:, -f_ny_old[1]:, -f_ny_old[2]:] = im_ft[-f_ny_old[0]:, -f_ny_old[1]:, -f_ny_old[2]:]
im_ft_pad = tf.manip.roll(im_ft, f_ny_old, axis=(0,1,2))
im_ft_pad = tf.pad(im_ft_pad, ((0, (upsam_factor-1) * im_ft.shape[0]), (0, (upsam_factor-1) * im_ft.shape[1]), (0, (upsam_factor-1) * im_ft.shape[2])) , 'constant')
im_ft_pad = tf.manip.roll(im_ft_pad, -f_ny_old, axis=(0,1,2))

im_upsam = tf.ifft3d(im_ft_pad)

with tf.Session() as sess:
    im_upsamc, im_ft_padc, im_ftc, imc = sess.run([im_upsam, im_ft_pad, im_ft, im])

print(np.abs(im_upsamc.imag).max())
print(np.sum(np.abs(im_upsamc.imag)))
print(np.abs(im_upsamc.real).max())
print(np.sum(np.abs(im_upsamc.real)))

fig, axes = plt.subplots(2,2)
axes[0,0].imshow(imc[...,7])
axes[0,1].imshow(np.log(np.abs(im_ftc[...,0])+1))
axes[1,0].imshow(np.log(np.abs(im_ft_padc[...,0])+1))
axes[1,1].imshow(im_upsamc.real[...,upsam_factor*7])

# %% numpy
###%% 3D
##
sh = np.array(np_im.shape)
np_im_ft = np.fft.fftn(np_im)
# this works but is not transferable to tf
#np_im_ft_pad = np.zeros(upsam_factor*sh, dtype=np.complex)
#np_im_ft_pad[:f_ny_old[0],  :f_ny_old[1],  :f_ny_old[2] ] = np_im_ft[:f_ny_old[0],  :f_ny_old[1],  :f_ny_old[2]]
#np_im_ft_pad[:f_ny_old[0],  :f_ny_old[1],  -f_ny_old[2]:] = np_im_ft[:f_ny_old[0],  :f_ny_old[1],  -f_ny_old[2]:]
#np_im_ft_pad[:f_ny_old[0],  -f_ny_old[1]:, :f_ny_old[2] ] = np_im_ft[:f_ny_old[0],  -f_ny_old[1]:, :f_ny_old[2]]
#np_im_ft_pad[:f_ny_old[0],  -f_ny_old[1]:, -f_ny_old[2]:] = np_im_ft[:f_ny_old[0],  -f_ny_old[1]:, -f_ny_old[2]:]
#np_im_ft_pad[-f_ny_old[0]:, :f_ny_old[1],  :f_ny_old[2] ] = np_im_ft[-f_ny_old[0]:, :f_ny_old[1],  :f_ny_old[2]]
#np_im_ft_pad[-f_ny_old[0]:, :f_ny_old[1],  -f_ny_old[2]:] = np_im_ft[-f_ny_old[0]:, :f_ny_old[1],  -f_ny_old[2]:]
#np_im_ft_pad[-f_ny_old[0]:, -f_ny_old[1]:, :f_ny_old[2] ] = np_im_ft[-f_ny_old[0]:, -f_ny_old[1]:, :f_ny_old[2]]
#np_im_ft_pad[-f_ny_old[0]:, -f_ny_old[1]:, -f_ny_old[2]:] = np_im_ft[-f_ny_old[0]:, -f_ny_old[1]:, -f_ny_old[2]:]
##print(np_im2d_ft_pad.shape)

np_im_ft_pad = np.roll(np_im_ft, f_ny_old, axis=(0,1,2))
np_im_ft_pad = np.pad(np_im_ft_pad, ((0, (upsam_factor-1) * np_im_ft.shape[0]), (0, (upsam_factor-1) * np_im_ft.shape[1]), (0, (upsam_factor-1) * np_im_ft.shape[2])) , 'constant')
np_im_ft_pad = np.roll(np_im_ft_pad, -f_ny_old, axis=(0,1,2))

np_im_upsam = np.fft.ifftn(np_im_ft_pad)

print(np.abs(np_im_upsam.imag).max())
print(np.sum(np.abs(np_im_upsam.imag)))
print(np.abs(np_im_upsam.real).max())
print(np.sum(np.abs(np_im_upsam.real)))

fig, axes = plt.subplots(2,2)
axes[0,0].imshow(np_im[...,7])
axes[0,1].imshow(np.log(np.abs(np_im_ft[...,0])+1))
axes[1,0].imshow(np.log(np.abs(np_im_ft_pad[...,0])+1))
axes[1,1].imshow(np_im_upsam.real[...,2*7])

# %% 2D
#np_im2d = np_im[...,8]
#
#upsam_factor = 2
#sh = np.array(np_im2d.shape)
#f_ny_old = sh//2
##print(f_ny_old)
#
#np_im2d_ft = np.fft.fft2(np_im2d)
## this works but is not transferable to tf
#np_im2d_ft_pad = np.zeros(upsam_factor*sh, dtype=np.complex)
##print(np_im2d_ft_pad.shape)
##np_im2d_ft_pad[:f_ny_old[0],:f_ny_old[1]] = np_im2d_ft[:f_ny_old[0],:f_ny_old[1]]
##np_im2d_ft_pad[:f_ny_old[0],-f_ny_old[1]:] = np_im2d_ft[:f_ny_old[0],-f_ny_old[1]:]
##np_im2d_ft_pad[-f_ny_old[0]:,:f_ny_old[1]:] = np_im2d_ft[-f_ny_old[0]:,:f_ny_old[1]]
##np_im2d_ft_pad[-f_ny_old[0]:,-f_ny_old[1]:] = np_im2d_ft[-f_ny_old[0]:,-f_ny_old[1]:]
#
#np_im2d_ft_pad = np.roll(np_im2d_ft, f_ny_old, axis=(0,1))
#np_im2d_ft_pad = np.pad(np_im2d_ft_pad, ((0, (upsam_factor-1) * np_im2d_ft.shape[0]), (0, (upsam_factor-1) * np_im2d_ft.shape[1])) , 'constant')
#np_im2d_ft_pad = np.roll(np_im2d_ft_pad, -f_ny_old, axis=(0,1))
#
#np_im2d_upsam = np.fft.ifft2(np_im2d_ft_pad)
#
##print(np.abs(np_im2d_upsam.imag).max())
##print(np.sum(np.abs(np_im2d_upsam.imag)))
##print(np.abs(np_im2d_upsam.real).max())
##print(np.sum(np.abs(np_im2d_upsam.real)))
#
#
#fig, axes = plt.subplots(2,2)
#axes[0,0].imshow(np_im2d)
#axes[0,1].imshow(np.log(np.abs(np_im2d_ft)+1))
#axes[1,0].imshow(np.log(np.abs(np_im2d_ft_pad)+1))
#axes[1,1].imshow(np_im2d_upsam.real)

# %% 1D
#l = 10  # seems to work for both even and odd
#harm = 4
#upsam_factor = 10
#
### FT of cos
##x = np.arange(l)*2*np.pi/l
##a = np.cos(harm*x)
##a_ft = np.fft.fft(a)
###print(np.real(a))
###print(np.sum(np.abs(np.imag(a_ft))))
###print(a_ft[harm])
###print(np.argmax(a_ft[0:l//2+1]))
##fig, axes = plt.subplots(2,1)
##axes[0].plot(x,a,'b-')
##axes[0].plot(x,a,'rx')
##axes[1].plot(np.real(a_ft), 'b-')
##axes[1].plot(np.real(a_ft), 'rx')
###plt.show()
#
### FT of delta
##a = np.zeros(l)
##a[harm] = 1
##a[len(a) - harm] = 1
##a_ft = np.fft.fft(a)
##fig, axes = plt.subplots(2,1)
##axes[0].plot(a,'b-')
##axes[0].plot(a,'rx')
##axes[1].plot(np.real(a_ft), 'b-')
##axes[1].plot(np.real(a_ft), 'rx')
###plt.show()
#
## Fourier upsampling
#x = np.arange(l)*2*np.pi/l
#a = np.cos(harm*x)
#a_ft = np.fft.fft(a)
#
## try even and odd input: (check!)
#f_ny_old = len(a_ft)//2
#
### -> a) fill new array at the right positions
##a_ft_pad = np.zeros(upsam_factor*l, dtype=np.complex)
##a_ft_pad[:f_ny_old+1] = a_ft[:f_ny_old+1]  # f_ny is same for pos and neg freqs
##a_ft_pad[-f_ny_old:] = a_ft[-f_ny_old:]
## -> b) pad current array and shift right half
#a_ft_pad = np.roll(a_ft, f_ny_old)
#a_ft_pad = np.pad(a_ft_pad, (0, (upsam_factor-1) * a_ft.shape[0]), 'constant')
#a_ft_pad = np.roll(a_ft_pad,-f_ny_old)
#
#a_upsam = np.fft.ifft(a_ft_pad)
#fig, axes = plt.subplots(4,1)
#axes[0].plot(a,'b-')
##axes[0].plot(a,'rx')
#axes[1].plot(a_ft.real, 'b-')
##axes[1].plot(np.real(a_ft), 'rx')
#axes[2].plot(a_ft_pad.real,'b-')
#axes[3].plot(a_upsam.real,'b-')
##axes[3].plot(a_upsam.real,'rx')
##plt.show()
#plt.tight_layout()