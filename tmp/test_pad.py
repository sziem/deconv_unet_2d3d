#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:16:31 2018

@author: Soenke
"""

import numpy as np

input_shape = (10, 101, 101, 101, 1)
network_depth = 4

# for same padding pad input with zeroes, st. input and output shapes match
# (problems might arise, when a layer has odd number of pixels in a dim)

im_shape = [input_shape[i] for i in [1,2,3]]
im_pad = [0, 0, 0]

print(im_shape)
print(im_pad)

#z_shape = input_shape[1] # channels last
#y_shape = input_shape[2]
#x_shape = input_shape[3]

#x_pad = 0
#y_pad = 0
#z_pad = 0

# applicable for all layers except first and last
#if x_shape % 2 == 0:
#    x_pad += 1

for j in range(len(im_shape)):  # z, y, x
    for i in range(network_depth-1):
        if im_shape[j] % 2 == 1:
            im_pad[j] += 2**i
        im_shape[j] = int(np.ceil(im_shape[j]/2))
        print(im_shape)
        print(im_pad)