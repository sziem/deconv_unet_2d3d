#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 14:40:45 2018

@author: soenke
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
import tensorflow as tf
#import scipy

# %% using numpy tricks(fast, only one copy during reshape)
def repeat2d(a, b0, b1):
    r, c = a.shape                                    # number of rows/columns
    rs, cs = a.strides                                # bytes to skip in each dim when traversing an array
    x = as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0)) # view a as 4D array
    return x.reshape(r*b0, c*b1)                      # create new 2D array

def repeat3d(a, b0, b1, b2):
    d, h, w = a.shape                                 # depth, height, width
    ds, hs, ws = a.strides                            # strides (values in stream) 
    x = as_strided(a, (d, b0, h, b1, w, b2), (ds, 0, hs, 0, ws, 0)) # view a as 6D array
    return x.reshape(d*b0, h*b1, w*b2)                # create new 2D array

# using builtin-numpy (copying 3 times)
def repeat3d_2(a, b0, b1, b2):
    return a.repeat(b0, axis=0).repeat(b1, axis=1).repeat(b2, axis=2)

def repeat3d_batch_2(a, b0, b1, b2):
    return a.repeat(b0, axis=1).repeat(b1, axis=2).repeat(b2, axis=3)

#a = np.array(np.arange(27).reshape([3,3,3]))
#print(a)
#
##print(a.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2))
#print(repeat3d(a, 2, 2, 2))
#print(repeat3d_2(a, 2, 2, 2))
#print(np.all(repeat3d(a, 2, 2, 2) == repeat3d_2(a, 2, 2, 2)))


# %% tensorflow
#tf_b = tf.keras.layers.UpSampling3D
#tf_a = tf.constant(a)

# TODO: make function that does all dims simultaneously!


# %%  This is the best! (from s

def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:
    
    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor

# %% own solution:

def tf_repeat_along_axis2d(t, repeats, axis):
    """designed to imitatate np.repeat(t, repeats, axis=axis) for 2-dim. t"""
    if axis == 0:
        return tf.transpose(tf_repeat_along_axis2d(tf.transpose(t), repeats, 1))
    elif axis == 1:
        return tf.reshape(tf.tile(tf.reshape(t, (-1, 1)), (1, repeats)), 
                          (t.get_shape()[0], -1))
    else:
        msg = ("axis " + str(axis) + " is out of bounds for array of "  + 
               "dimension " + str(len(t.get_shape())))
        raise IndexError(msg)

def tf_repeat_along_axis3d(t, repeats, axis):
    """designed to imitatate np.repeat(t, repeats, axis=axis) for 3-dim. t"""
    if axis == 0:
        return tf.transpose(tf_repeat_along_axis3d(tf.transpose(t), repeats, 2))
    elif axis == 1:
        return tf.transpose(tf_repeat_along_axis3d(tf.transpose(t, (2,0,1)), repeats, 2), (1,2,0))
    elif axis == 2:
        return tf.reshape(tf.tile(tf.reshape(t, (-1, 1)), (1, repeats)), 
                          (t.get_shape()[0], t.get_shape()[1], -1))
    else:
        msg = ("axis " + str(axis) + " is out of bounds for array of "  + 
               "dimension " + str(len(t.get_shape())))
        raise IndexError(msg)

def tf_repeat_along_axis5d(t, repeats, axis):
    """designed to imitatate np.repeat(t, repeats, axis=axis) for 5-dim t"""
    if axis == 0:
        return tf.transpose(tf_repeat_along_axis5d(tf.transpose(t), repeats, 4)) # v
    elif axis == 1:
        return tf.transpose(tf_repeat_along_axis5d(tf.transpose(t, (2,3,4,0,1)), repeats, 4), (3,4,0,1,2))
    elif axis == 2:
        return tf.transpose(tf_repeat_along_axis5d(tf.transpose(t, (3,4,0,1,2)), repeats, 4), (2,3,4,0,1))
    elif axis == 3:
        return tf.transpose(tf_repeat_along_axis5d(tf.transpose(t, (4,0,1,2,3)), repeats, 4), (1,2,3,4,0))  # v
    elif axis == 4:
        return tf.reshape(tf.tile(tf.reshape(t, (-1, 1)), (1, repeats)), 
                          (t.get_shape()[0], t.get_shape()[1], 
                           t.get_shape()[2], t.get_shape()[3], -1)) # v
    else:
        msg = ("axis " + str(axis) + " is out of bounds for array of "  + 
               "dimension " + str(len(t.get_shape())))
        raise IndexError(msg)

def tf_repeat3d(t, r0, r1, r2):
    """
    calls tf_repeat_along_axis3d 3 times to achieve NN upsampling.
    t is a tf-tensor
    r0, r1 and r2 are repeats in axis 0, 1 and 2 respectively.
    """
    return tf_repeat_along_axis3d(
                tf_repeat_along_axis3d(
                        tf_repeat_along_axis3d(t, r0, 0), r1, 1), r2, 2)
    
def tf_repeat3d_batch(t, rd, rh, rw, data_format='channels_last',
                                   name='NN_upsampling'):
    """
    calls tf_repeat_along_axis5d 3 times to achieve NN upsampling.
    t is a tf-tensor with shape (n_batches, depth, height, width, channels)  
    ("channels last!")
    rd, rh and rw are repeats in depth, height and width respectively.
    """
    return tf_repeat_along_axis5d(
                tf_repeat_along_axis5d(
                        tf_repeat_along_axis5d(t, rd, 1), rh, 2), rw, 3)


# %% testing


np_x = np.array(np.arange(54).reshape([2,3,3,3,1]))  # 5d tensor as in CNN
#print(x)
r1, r2, r3 = 2, 2, 2  # repeats in z, y, x
#print(np.repeat(np_x, 2, axis=axis))
print(repeat3d_2(np_x, r1, r2, r3))

print("=================================================================")
x = tf.constant(np_x)
with tf.Session() as sess:
    # print(sess.run(tf_repeat(x, [1,2,2,2,1])))
    #print(sess.run(tf_repeat5d(x, r0, r1, r2)))
    #print(np.all(np.repeat(np_x, 2, axis=axis) == sess.run(tf_repeat_along_axis5d(x, 2, axis=axis))))
    print(np.all(np.repeat(np.repeat(np.repeat(np.repeat(np.repeat(np_x, 1, 0), 2, 1), 2, 2), 2, 3), 1, 4) == 
                 sess.run(tf_repeat(x, [1,2,2,2,1]))))
