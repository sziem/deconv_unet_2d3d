#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 15:28:08 2018

@author: soenke
"""

import numpy as np
import tensorflow as tf
import functools
import warnings

# TODO: add name_scopes

def integer_warning(func):
    """This is a decorator which can be used to mark functions
    that don't automatically use floating point ops on integers. 
    It will result in a warning being emitted
    when the function is used with integers."""
    # TODO: why is result printed first then warning below?
    # TODO: deprecation warning in tf
    # TODO: how to enable it for arbitrary X? like this?
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        integer_dtypes = [tf.int8, tf.int16, tf.int32, tf.int64, 
                          tf.uint8, tf.uint16, tf.uint32, tf.uint64]
        if X.dtype in integer_dtypes:
            warnings.simplefilter('always', Warning)  # turn off filter
            warnings.warn("X is an" + str(X.dtype) + 
                          "mean and std are calculated as integers.")
                          #". {} has done integer ops.".format(func.__name__))
            warnings.simplefilter('default', Warning)  # reset filter
        return func(*args, **kwargs)
    return new_func


# %% called "batch" because they are intended to be used on 5d-batches
# within the tf-workflow.  These approximate the norm by using values 
# from entire batch
#
# TODO -- not implemented
#
#def minmax_scale_entire_batch(X, target_range=(0,1)):
#    """    
#    Transforms all images in batch by scaling each to a given range.
#    
#    Args:
#        X : tensor, shape (n_image, depth, height, width, channel)
#            The data.
#        target_range : tuple (min, max), default=(0, 1)
#            Desired range of transformed data.
#    
#    Returns:
#        X_scaled: tensor, same shape as input
#            Scaled data.
#    
#    Scales and translates each image such that the minimum and maximum are
#    given by range.
#
#    The transformation is:
#
#        im_std = (im - im.min() / (im.max() - im.min())
#        im_scaled = im_std * (max - min) + min
#            
#    for each image in the batch.
#    
#    method naming follow sklearn.preprocessing, but has been adapted to 3d
#    image stacks, i.e. 5d-data
#    """
#    t_min, t_max = target_range  # t = target, c=current
#    # uses broadcasting trick
#    c_min = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(
#            tf.reduce_min(X,(1,2,3,4)), -1), -1), -1), -1)
#    c_max = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(
#            tf.reduce_max(X,(1,2,3,4)), -1), -1), -1), -1)
#
#    X_std = (X - c_min) / (c_max - c_min)
#    return X_std * (t_max - t_min) + t_min
#
#
#def maxabs_scale_entire_batch(X):
#    """
#    Scales each image individually such that the maximal absolute value 
#    of each image in the batch will be 1.0.
#
#    Args:
#        X : tensor, shape (n_samples, n_features)
#            The data.
#
#    Returns:
#        X_scaled : tensor, shape (n_samples, n_features)
#            Scaled data.
#
#    The resulting range generally is [-1, 1] by dividing by the max abs.
#    For images with all pos values, this will scale to [0, 1].
#    """
#    # c = current
#    c_maxabs = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(
#            tf.reduce_max(tf.abs(X),(1,2,3,4)), -1), -1), -1), -1)
#    return X / c_maxabs
#
#
#def subtract_min_entire_batch(X):
#    """    
#    Transforms all images in batch by subtracting their respective min from 
#    themselves.  This was mainly done to test the implementation and not
#    for a specific use-case, but maybe it is useful later
#    
#    Args:
#        X : tensor, shape (n_image, depth, height, width, channel)
#            The data.
#    
#    Returns:
#        X_subtr: tensor, same shape as input
#            data where min has been subtracted.
#
#    The transformation is:
#
#        im_std = im - im.min()
#    
#    for each image in the batch.
#    """
#    return X - tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(
#            tf.reduce_min(X,(1,2,3,4)), -1), -1), -1), -1)
#
#
##def normalize_entire_batch(X):
##    """
##    This would normalize each image to a sum of 1.
##    This does not seem useful for image data, though.
##    """
##    pass
#
#@integer_warning
#def subtract_mean_entire_batch(X):
#    # c = current
#    c_means = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(
#            tf.reduce_mean(X,(1,2,3,4)), -1), -1), -1), -1)
#    return X - c_means
#
#@integer_warning
#def std_scale_entire_batch(X):
##    integer_dtypes = [tf.int8, tf.int16, tf.int32, tf.int64, 
##                  tf.uint8, tf.uint16, tf.uint32, tf.uint64]
##    if X.dtype is in integer_dtypes:
##        warn("X is an", X.dtype, ". std_scale_batch will do integer ops.")
#    # c = current
#    c_stds = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(
#            reduce_std(X,(1,2,3,4)), -1), -1), -1), -1)
#    return X / c_stds


# %% called "batch" because they are intended to be used on 5d-batches
# within the tf-workflow.  These operate on each image individually
# "dataset" versions function the same but use numpy
def minmax_scale_individually_batch(X, target_range=(0,1)):
    """    
    Transforms all images in batch by scaling each to a given range.
    
    Args:
        X : tensor, shape (n_image, depth, height, width, channel)
            The data.
        target_range : tuple (min, max), default=(0, 1)
            Desired range of transformed data.
    
    Returns:
        X_scaled: tensor, same shape as input
            Scaled data.
    
    Scales and translates each image such that the minimum and maximum are
    given by range.

    The transformation is:

        im_std = (im - im.min() / (im.max() - im.min())
        im_scaled = im_std * (max - min) + min
            
    for each image in the batch.
    
    method naming follow sklearn.preprocessing, but has been adapted to 3d
    image stacks, i.e. 5d-data
    """
    t_min, t_max = target_range  # t = target, c=current
    # uses broadcasting trick
    c_min = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(
            tf.reduce_min(X,(1,2,3,4)), -1), -1), -1), -1)
    c_max = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(
            tf.reduce_max(X,(1,2,3,4)), -1), -1), -1), -1)

    X_std = (X - c_min) / (c_max - c_min)
    return X_std * (t_max - t_min) + t_min


def maxabs_scale_individually_batch(X):
    """
    Scales each image individually such that the maximal absolute value 
    of each image in the batch will be 1.0.

    Args:
        X : tensor, shape (n_samples, n_features)
            The data.

    Returns:
        X_scaled : tensor, shape (n_samples, n_features)
            Scaled data.

    The resulting range generally is [-1, 1] by dividing by the max abs.
    For images with all pos values, this will scale to [0, 1].
    """
    # c = current
    c_maxabs = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(
            tf.reduce_max(tf.abs(X),(1,2,3,4)), -1), -1), -1), -1)
    return X / c_maxabs


#def subtract_min_individually_batch(X):
#    """    
#    Transforms all images in batch by subtracting their respective min from 
#    themselves.  This was mainly done to test the implementation and not
#    for a specific use-case, but maybe it is useful later
#    
#    Args:
#        X : tensor, shape (n_image, depth, height, width, channel)
#            The data.
#    
#    Returns:
#        X_subtr: tensor, same shape as input
#            data where min has been subtracted.
#
#    The transformation is:
#
#        im_std = im - im.min()
#    
#    for each image in the batch.
#    """
#    return X - tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(
#            tf.reduce_min(X,(1,2,3,4)), -1), -1), -1), -1)


#def normalize_individually_batch(X):
#    """
#    This would normalize each image to a sum of 1.
#    This does not seem useful for image data, though.
#    """
#    pass

@integer_warning
def subtract_mean_individually_batch(X):
    # c = current
    c_means = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(
            tf.reduce_mean(X,(1,2,3,4)), -1), -1), -1), -1)
    return X - c_means

@integer_warning
def std_scale_individually_batch(X):
    # c = current
    c_stds = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(
            reduce_std(X,(1,2,3,4)), -1), -1), -1), -1)
    return X / c_stds


# these are not implemented in tf-core.
#def reduce_var(x, axis=None, keepdims=False):
#    """Variance of a tensor, alongside the specified axis.
#
#    # Arguments
#        x: A tensor or variable.
#        axis: An integer, the axis to compute the variance.
#        keepdims: A boolean, whether to keep the dimensions or not.
#            If `keepdims` is `False`, the rank of the tensor is reduced
#            by 1. If `keepdims` is `True`,
#            the reduced dimension is retained with length 1.
#
#    # Returns
#        A tensor with the variance of elements of `x`.
#    """
#    # TODO: this does integer ops
#    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
#    devs_squared = tf.square(x - m)
#    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)
#
def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    # sqrt will not work on ints
    # TODO: use isinstance
    dt = x.dtype
    allowed_dtypes = [tf.bfloat16, tf.float16, tf.float32, tf.float64, 
                      tf.complex64, tf.complex128]
    if dt not in allowed_dtypes:
        x = tf.to_float(x)
    mean, std = tf.nn.moments(x, axis)
    return tf.cast(tf.sqrt(std), dt)


# %% called "dataset" because they are intended to be used on the entire 
# 5d-dataset and use numpy.
# "batch" versions function the same but use tf
def minmax_scale_dataset(X, target_range=(0,1)):
    """
    Transforms all images in dataset by scaling each to a given range.
    
    Args:
        X : array-like, shape (n_image, depth, height, width, channel)
            The data.
        target_range : tuple (min, max), default=(0, 1)
            Desired range of transformed data.
    
    Returns:
        X_scaled: array-like, same shape as input
            Scaled data.
    
    Scales and translates each image such that the minimum and maximum are
    given by range.

    The transformation is:

        im_std = (im - im.min() / (im.max() - im.min())
        im_scaled = im_std * (max - min) + min
            
    for each image in the dataset.
    
    method naming follow sklearn.preprocessing, but has been adapted to 3d
    image stacks, i.e. 5d-data
    """
    
    t_min, t_max = target_range  # t = target, c=current
    # uses broadcasting trick
    c_min = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(
            X.min((1,2,3,4)), -1), -1), -1), -1)
    c_max = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(
            X.max((1,2,3,4)), -1), -1), -1), -1)

    X_std = (X - c_min) / (c_max - c_min)
    return X_std * (t_max - t_min) + t_min


def maxabs_scale_dataset(X):
    """
    Scales each image individually such that the maximal absolute value 
    of each image in the dataset will be 1.0.

    Args:
        X : array-like, shape (n_samples, n_features)
            The data.

    Returns:
        X_scaled : array-like, shape (n_samples, n_features)
            Scaled data.

    The resulting range generally is [-1, 1] by dividing by the max abs.
    For images with all pos values, this will scale to [0, 1].
    """
    # c = current
    c_maxabs = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(
            np.abs(X).max((1,2,3,4)), -1), -1), -1), -1)
    return X / c_maxabs


def subtract_min_dataset(X):
    """    
    Transforms all images in dataset by subtracting their respective min from 
    themselves.  This was mainly done to test the implementation and not
    for a specific use-case, but maybe it is useful later
    
    Args:
        X : array-like, shape (n_image, depth, height, width, channel)
            The data.
    
    Returns:
        X_subtr: array-like, same shape as input
            data where min has been subtracted.

    The transformation is:

        im_std = im - im.min()
    
    for each image in the dataset.
    """
    return X - np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(
            X.min((1,2,3,4)), -1), -1), -1), -1)


#def normalize_dataset(X):
#    """
#    This would normalize each image to a sum of 1.
#    This does not seem useful for image data, though.
#    """
#    pass

def subtract_mean_dataset(X):
    # c = current
    c_means = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(
            np.mean(X,(1,2,3,4)), -1), -1), -1), -1)
    return X - c_means

def std_scale_dataset(X):
    # c = current
    c_stds = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(
            np.std(X,(1,2,3,4)), -1), -1), -1), -1)
    return X / c_stds



# %% testing
#X = np.arange(36).reshape((2,2,3,3,1))
#np_X = np.arange(72).reshape((2,4,3,3,1))
#np_X[1] = 2*np_X[1]
##print(np_X)
#X = tf.constant(np_X, dtype=tf.float32)  
#with tf.Session() as sess:
    #print(sess.run(subtract_min_batch(X)))
    #print(sess.run(minmax_scale_batch(X)))
    #print(sess.run(maxabs_scale_batch(X)))
    #print(sess.run(subtract_mean_batch(X)))
    #print(sess.run(std_scale_batch(tf.to_float(X))))
#    print(sess.run(tf.reduce_mean(X, (1,2,3,4))))
#    print(sess.run(tf.reduce_mean(subtract_mean_batch(X), (1,2,3,4))))
#    print(sess.run(tf.reduce_mean(std_scale_batch(subtract_mean_batch(X)), (1,2,3,4))))
#    print(sess.run(reduce_std(X, (1,2,3,4))))
#    print(sess.run(reduce_std(std_scale_batch(X), (1,2,3,4))))
#    print(sess.run(reduce_std(std_scale_batch(subtract_mean_batch(X)), (1,2,3,4))))

#print(np.max(np_X, axis = 0))
#print(np.expand_dims(np.expand_dims(np.min(np.min(np_X, axis=-1),axis=-1), -1), -1))
#print(subtract_min_dataset(np_X))
#print(minmax_scale_dataset(np_X))
#print(maxabs_scale_dataset(np_X))
#print(subtract_mean_dataset(np_X))
#print(std_scale_dataset(np_X))
#print(np.mean(np_X, (1,2,3,4)))
#print(np.mean(subtract_mean_dataset(np_X), (1,2,3,4)))
#print(np.std(np_X, (1,2,3,4)))
#print(np.std(std_scale_dataset(np_X), (1,2,3,4)))
#print(np.std(std_scale_dataset(subtract_mean_dataset(np_X)), (1,2,3,4)))
#print(np.mean(std_scale_dataset(subtract_mean_dataset(np_X)), (1,2,3,4)))
    
    
    
# %% lower than 5d
    
#def minmax_scale_dataset1d(X, target_range=(0,1)):
#    t_min, t_max = target_range  # t = target
#    # uses broadcasting trick
#    c_min = np.expand_dims(X.min(axis=1), -1) # c = current
#    c_max = np.expand_dims(X.max(axis=1), -1) # c = current
#
#    X_std = (X - c_min) / (c_max - c_min)
#    return X_std * (t_max - t_min) + t_min
#
#def minmax_scale_dataset2d(X, target_range=(0,1)):
#    t_min, t_max = target_range  # t = target
#    # uses broadcasting trick
#    c_min = np.expand_dims(np.expand_dims(X.min(-1).min(-1), -1), -1)
#    c_max = np.expand_dims(np.expand_dims(X.max(-1).max(-1), -1), -1)
#
#    X_std = (X - c_min) / (c_max - c_min)
#    return X_std * (t_max - t_min) + t_min
#
#def minmax_scale_dataset3d(X, target_range=(0,1)):
#    t_min, t_max = target_range  # t = target
#    # uses broadcasting trick
#    c_min = np.expand_dims(np.expand_dims(np.expand_dims(
#                                X.min(-1).min(-1).min(-1), -1), -1), -1)
#    c_max = np.expand_dims(np.expand_dims(np.expand_dims(
#                                X.max(-1).max(-1).max(-1), -1), -1), -1)
#
#    X_std = (X - c_min) / (c_max - c_min)
#    return X_std * (t_max - t_min) + t_min
#
#def subtract_min_dataset1d(X):
#    return X - np.expand_dims(X.min(axis=1), -1)
#
#def subtract_min_dataset2d(X):
#    # TODO: careful: will not warn if input has wrong shape
#    return X - np.expand_dims(np.expand_dims(X.min(-1).min(-1), -1), -1)
#
#def subtract_min_dataset3d(X):
#    # TODO: careful: will not warn if input has wrong shape (e.g. 2d)
#    return X - np.expand_dims(np.expand_dims(np.expand_dims(
#                                X.min(-1).min(-1).min(-1), -1), -1), -1)
