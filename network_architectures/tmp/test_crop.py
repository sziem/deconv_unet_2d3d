# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:41:01 2018

@author: photon
"""

import tensorflow as tf
import numpy as np
import operator

def convert_shape_to_np_array(sh):
    """converts list, tuple, ... and also tf.TensorShape to np-array"""
    if isinstance(sh, tf.TensorShape):
        sh = sh.as_list()
    return np.array(sh)
#
#def crop(t, new_shape):  # DBG?
#    """
#    will crop half the pixels on the left and half on the right of t st. it
#    has new_shape. If the difference in shapes is odd in some dim, it will 
#    crop the extra pixel on the right.
#    
#    Ignores any dimensions that have been set to None in either new_shape or t.
#    """
#    # TODO: better: if shape is none in t and not none in new shape, then try
#    # to set it to new_shape  --> but that can only work in runtime
#    
#    input_shape = convert_shape_to_np_array(t.shape)
#    new_shape = convert_shape_to_np_array(new_shape)
#
#    assert len(input_shape) == len(new_shape)
#    ndims = len(new_shape)
#    
#    masks = list()
#    for i in range(ndims):
#        mask_i = np.zeros(input_shape[i], dtype=np.bool)
#        mask_i[:new_shape[i]] = True
#        mask_i = np.roll(mask_i, (input_shape[i] - new_shape[i]) //2)
#        # for broadcasting
#        for j in range(i):
#            mask_i = np.expand_dims(mask_i, 0)
#        for j in range(i+1, ndims):
#            mask_i = np.expand_dims(mask_i, -1)
#        masks.append(mask_i)
#    mask = np.prod(masks)  # test
#    
##    with tf.name_scope("crop"):
#    out = tf.boolean_mask(t, mask, name="crop")
#    return out
#
#t = tf.zeros((32,25))
#new_shape= (30, 24)
#op = crop(t, new_shape)
#print(tf.Session().run(op).shape)

def crop_middle(t, new_shape):
    """
    will crop half the pixels on the left and half on the right of t st. it
    has new_shape. If the difference in shapes is odd in some dim, it will 
    crop the extra pixel on the right/bottom/... .
    
    works for n-dimensional tensors
    
    source: https://stackoverflow.com/a/50322574
    
    Ignores any dimensions that is None in either new_shape or t.
    """
    input_shape = convert_shape_to_np_array(t.shape)
    new_shape = convert_shape_to_np_array(new_shape)

    # detect Nones
    is_none_in = input_shape == np.array(None)
    is_none_new = new_shape == np.array(None)
    is_none = is_none_in + is_none_new
    
    not_none = np.logical_not(is_none)
    if len(input_shape) != len(new_shape):
        raise ValueError("new_shape must have same number of dims as t.\n" +
                         "len(t.shape) is " + str(len(t.shape)) + ".\n" +
                         "len(new_shape) is " + str(len(new_shape)) + ".")
    if np.any(new_shape[not_none] > input_shape[not_none]):
        raise ValueError("All entries of new_shape must be smaller than t.shape.\n" + 
                         "t.shape is " + str(input_shape) + ".\n" +
                         "new_shape is " + str(new_shape) + ".")
    if np.any(new_shape[not_none] < 0):
        raise ValueError("All entries of new_shape must be greater than 0.\n" +
                         "new_shape is " + str(new_shape) + ".")
    
    # slices are created here
    starts = np.zeros_like(input_shape, dtype=object) # dtype in case of None
    ends = np.zeros_like(starts)
    for i, ignore in enumerate(is_none):
        if not ignore:
            starts[i] = (input_shape[i]-new_shape[i])//2
            ends[i] = starts[i] + new_shape[i]
        else:
            starts[i] = None
            ends[i] = None

    slices = tuple(map(slice, starts, ends))
    with tf.name_scope("crop"):
        return t[slices]


def crop_right(t, new_shape):
    """
    will crop pixels on the right of t st. it has new_shape. 
    There was a TODO-comment to implement this, but I don't remember what for.
    Usually prefer central_crop
    
    Works for n-dimensional tensors
    
    Ignores any dimensions that is None in either new_shape or t.
    """
    input_shape = convert_shape_to_np_array(t.shape)
    new_shape = convert_shape_to_np_array(new_shape)

    # detect None
    is_none_in = input_shape == np.array(None)
    is_none_new = new_shape == np.array(None)
    is_none = is_none_in + is_none_new
    
    not_none = np.logical_not(is_none)
    if len(input_shape) != len(new_shape):
        raise ValueError("new_shape must have same number of dims as t.\n" +
                         "len(t.shape) is " + str(len(t.shape)) + ".\n" +
                         "len(new_shape) is " + str(len(new_shape)) + ".")
    if np.any(new_shape[not_none] > input_shape[not_none]):
        raise ValueError("All entries of new_shape must be smaller than t.shape.\n" + 
                         "t.shape is " + str(input_shape) + ".\n" +
                         "new_shape is " + str(new_shape) + ".")
    if np.any(new_shape[not_none] < 0):
        raise ValueError("All entries of new_shape must be greater than 0.\n" +
                         "new_shape is " + str(new_shape) + ".")
    
    # slices are created here
    starts = np.zeros_like(input_shape, dtype=object) # dtype in case of None
    ends = np.zeros_like(starts)
    for i, ignore in enumerate(is_none):
        if not ignore:
            starts[i] = 0
            ends[i] = starts[i] + new_shape[i]
        else:
            starts[i] = None
            ends[i] = None

    slices = tuple(map(slice, starts, ends))
    with tf.name_scope("crop"):
        return t[slices]  


np_img = np.arange(25).reshape(5,5)
img = tf.constant(np_img)
#new_shape = (0,0) #ok -> empty with shape (0,0)
#new_shape = (2,0) #ok -> empty with shape (2,0)
#new_shape = (4,4) #ok -> crops more on the right
#new_shape = (5,5) #ok -> returns original array
#new_shape = (6,5) # > caught as exception. Otherwise would return entry -1 etc.
#new_shape = (4,)  # -> caught as exception.  Otherwise leaves 5 as second entry.
#new_shape = (-2,-2)
new_shape = (2,2)

img_crop = crop_right(img, new_shape)
with tf.Session():
    np_img_crop = img_crop.eval()

print(np_img.shape)
print(np_img)
print(np_img_crop)
print(np_img_crop.shape)
