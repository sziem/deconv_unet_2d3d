# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:43:36 2018

@author: photon
"""

import numpy as np
import h5py
import os
import scipy.io
import random
import tensorflow.keras.utils as kutils

# Data IO into numpy



# transform array/list of indices to one_hot array
# https://stackoverflow.com/a/49217762
def one_hot(indices, num_classes, axis):
    """returns one-hot array.  hot==1, not==0"""
    # TODO: axis
    onehot = kutils.to_categorical(indices, num_classes=num_classes)
    np.swapaxes(onehot, axis, -1)
    return onehot


# %% generic wrappers
#def np_load_pair(x_fname, y_fname, **kwargs):
#    x = np_load_single(os.path.join(pair_path, x_fname, **kwargs))
#    y = np_load_single(os.path.join(pair_path, y_fname, **kwargs))
#    return x, y

# TODO: would it be nice to use input_format and output_format strings instead 
# of add_channel_dim? (eg. input format: 'hw' -> output format 'hwc'
def np_load(fpath, expand_dims=None, **kwargs):
    """
    wrapper for file input into numpy-arrays.
    
    Args:
        fpath (str) : file path.  full path to file including extension.
        expand_dims (None or int) : Add an empty dim using expand_dims.
            Defaults to None (don't add any).

    Returns:
        data (numpy-array)
        
    List of kwargs:
        For "mat"-files:
            field (str) : name of the saved matlab variable.
                Defaults (None) to the file name, e.g. 'data.mat' -> field='data'
            version (str) : mat-file-version. ('v7' or 'v7.3')
                Defaults to 'v7.3', a hdf5-derivative
        For "npy"-files:
            kwargs to np.load
    """
    ftype = os.path.splitext(fpath)[1][1:] # extension without dot
    
    # check if filetype is supported
    allowed_filetypes = ["mat", "npy"]
    if ftype not in allowed_filetypes :
        raise ValueError(
                "Implemented filetypes are " + str(allowed_filetypes) + ". " +
                "You provided fpath " + fpath + "with detected extension " + 
                ftype + ".")
    # return
    if ftype == "mat":
        # kwargs: field, version
        arr = load_mat(fpath, **kwargs)
    elif ftype == "npy":
        # kwargs to np.load
        arr = load_npy(fpath, **kwargs)
    else:  # this case should never arise because of the check above.
        raise ValueError("file type " + ftype +  " is not supported.")
        
    if expand_dims is not None:
        arr = np.expand_dims(arr, expand_dims)
        
    return arr


# %% load mat
def load_mat(filename, field=None, version='v73'):
    """
    Load a mat-file generated with matlab's save-function like
    "save('data', 'data', version)".
    
    Args:
        filename (str) : should end with ".mat"; can be absolute or relative 
            path.
        field (str) : name of the saved matlab variable.
            Defaults (None) to the file name, e.g. 'data.mat' -> field='data'
        version (str) : mat-file-version.
            Defaults to 'v73', a hdf5-derivative
            
    Returns:
        data (numpy-array)
    
    Notes:
        - By convention, I always save the 'data' as 'data.mat'.  This will not 
          work, if the data is saved under a different name. Supply 'field' in 
          that case.
        - wrapper for load_mat_v7 and load_mat_v73
    """
    # TODO: try version v7 except NotImplementedError then do v73
    if version in ['73', '7.3', 'v73', 'v7.3', '-v73', '-v7.3']:
        return load_mat_v73(filename=filename, field=field)
    elif version in ['7', 'v7', '-v7']:
        return load_mat_v7(filename=filename, field=field)

def load_mat_v73(filename, field=None):
    """
    Load a v7.3 mat-file generated with matlab's save-function like
    "save('data', 'data', '-v73')".
    
    Args:
        filename (str) : should end with ".mat"; can be absolute or relative 
            path.
        field (str) : name of the saved matlab variable.
            Defaults (None) to the file name, e.g. 'data.mat' -> field='data'
            
    Returns:
        data (numpy-array)
    
    By convention, I always save the 'data' as 'data.mat'.  This will not work,
    if the data is saved under a different name. Supply 'field' in that case
    """
    if field is None:
        # TODO change this to try block and give better error message
        fpath, fname = os.path.split(filename)
        field = fname[0:-4]  # without .mat
        #name given in matlab - 'imgData' is standard name at ukj
    with h5py.File(filename, 'r') as f:
        data = f[field].value 
    return np.array(data)

def load_mat_v7(filename, field=None):
    """
    Load a v7 (not v7.3!) mat-file generated with matlab's save-function like
    "save('data', 'data', '-v7')".
    
    Args:
        filename (str) : should end with ".mat"; can be absolute or relative 
            path.
        field (str) : name of the saved matlab variable.
            Defaults (None) to the file name, e.g. 'data.mat' -> field='data'
    
    Returns:
        data (numpy-array)
    
    Notes:
        - By convention, I always save the 'data' as 'data.mat'.  This will not 
          work, if the data is saved under a different name.
        - Might also work with older mat files.
    """
    # example: im = io.loadmat("im.mat")["im"] #[0][0][0] if saved from dip_image
    if field is None:
        # TODO change this to try block and give better error message
        fpath, fname = os.path.split(filename)
        field = fname[0:-4]  # without .mat
    return scipy.io.loadmat(filename)[field]


# %% load npy
def load_npy(filename):
    return np.load(filename)

# %% load txt
def loadtxt(filename):
    return np.loadtxt(filename)

def label_to_image(label, shape, dtype=None):
    return label*np.ones(shape, dtype=dtype)

# %% load entire datasets into numpy
    
# from h5
def _load_h5(filename, field):
    with h5py.File(filename, 'r') as data_file:
        X = np.array(data_file[field])
    return X

# from filelist
def _load_filelist(filelist, load_fn):
    """
    Loads all images in filelist and stacks concatenates them along a new
    first dimension (the n_images or batch_dimension of the dataset) resutling
    in a numpy-array with 
    dimensions (n_images, ...spatial_dims..., channels).
    
    All images in filelist should have the same shape and number of channels
    
    Args:
        filelist (list): list of strings with paths to files
        load_fn (function) : function which loads an individual file as 
            np-array.  load_fn should have one argument (file_path) and return
            one np-array.
    
    Returns:
        X, Y (numpy-arrays) with dimensions 
        (n_images, ...spatial_dims..., channels) corresponding to images 
        and groundtruth objects.
    """
    n_images = len(filelist)
    im_shape = load_fn(filelist[0]).shape
    shape = (n_images,) + im_shape
    dtype = load_fn(filelist[0]).dtype

    # load data into 5d-numpy-arrays
    # X contains noisy images, Y ground-truth images (a.k.a. objects)
    arr = np.zeros(shape, dtype=dtype)
    for i, filepath in enumerate(filelist): 
        arr[i] = load_fn(filelist[i])
        print("loading image", i, "/", n_images-1, "into np-array.")
    return arr  


#def check_data_shape(X, Y):
#    """check if X (all images) and Y (all objects) have same shape."""
#    if X.shape != Y.shape:
#        msg = ("X and Y must have same shape.\n" +
#              "X has shape" + str(X.shape) +".\n" +
#              "Y has shape" + str(Y.shape) + ".")
#        raise ValueError(msg)
    

# %% get generator
def _generator_from_h5(filename, field):
    with h5py.File(filename, 'r') as data_file:
        # TODO: what happens if there is a crash here
        for im in data_file[field]:
            yield im

# this is not well tested
# it is not possible to change seed within generator
# random.seed(2)
def _random_generator_from_h5(filename, field):
    with h5py.File(filename, 'r') as data_file:
        # TODO: what happens if there is a crash in here?
        n_images = sum(_ for im in data_file[field])
        indexlist = list(range(n_images))
        random.shuffle(indexlist)  # shuffles in place
        for i in indexlist:
            yield data_file[field][i]
