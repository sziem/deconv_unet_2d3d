#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 15:28:18 2018

@author: soenke
"""

import numpy as np
import os
import os.path as path
import h5py
import scipy.io as io
import json
from warnings import warn
from toolbox.decorators import deprecated


# %% create h5-dataset

def create_datasets(np_X_train, np_Y_train, np_X_test, np_Y_test, 
                      target_path, name="vascu_pairs"):
    """
    Fill a h5-file with X and Y to create the dataset.
    
    Args:
        np_X_train, np_Y_train, np_X_test, np_Y_test (numpy-arrays): 
            images and groundtruth with dimensions 
            (n_images, depth, height, width, channels)
        target_path (str): 
            path (absolute or relative from current working 
            directory) to which h5-file is written.
        name (str): name of output files
    
    Creates two h5-file "name"_train.h5 and "name"_test.h5.
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    _create_train_dataset(np_X_train, np_Y_train, name, target_path)
    _create_test_dataset(np_X_test, np_Y_test, name, target_path)
    # return ?

def _create_train_dataset(np_X_train, np_Y_train, name, target_path):
    return _create_h5_dataset(np_X_train, np_Y_train, name+"_train.h5", target_path)

def _create_test_dataset(np_X_test, np_Y_test, name, target_path):
    return _create_h5_dataset(np_X_test, np_Y_test, name+"_test.h5", target_path)

def _create_h5_dataset(np_X, np_Y, name, target_path):
    sh = np_X.shape
    print(name, "shape:", sh)
    n_images = sh[0]
    with h5py.File(os.path.join(target_path, name), 'w') as dataset_file:
        dataset_file.create_dataset('X', sh, dtype='f')  # noisy images
        dataset_file.create_dataset('Y', sh, dtype='f')  # ground truth objects    
        for k in range(n_images):
            dataset_file['X'][k] = np_X[k].astype('float32')  # fill hdf5-file
            dataset_file['Y'][k] = np_Y[k].astype('float32')  # fill hdf5-file
        dataset_file.close()  # TODO is this necessary?
    # return ?


# %% load datasets
@deprecated("This is a modified copy of load_vascu_synth_data that deals " +
            "with the images that were split from the too-big images. " +
            " load_vascu_synth_data should be preferred.")
def load_vascu_synth_data_split(directory, dataset_specs="dataset_specs.json"):
    """
    THIS IS A MODIFIED COPY OF LOAD_VASCU_SYNTH_DATA THAT DEALS WITH THE 
    IMAGES THAT WERE SPLIT FROM THE IMAGES THAT WERE TOO BIG.
    
    Loads the images forming from my datasets generated with vascu_synth and
    matlab and returns them as 5d-numpy-array with dimensions 
    (n_images, depth, height, width, channels).
    
    Args:
        directory (str): path to the dataset
        dataset_specs (str): path to json-file (relative from directory) 
                             specifying dataset structure
    
    Returns:
        X_train, Y_train, X_test, Y_test (numpy-arrays) with dimensions 
        (n_images, depth, height, width, channels) corresponding to images 
        and groundtruth objects.
    
    A sample vascu-synth dataset is available from http://vascusynth.cs.sfu.ca/).
    
    My modified version contains groundtruth images ("obj") and corresponding 
    simulated images ("im") as mat-files.  The enclosing folder structure 
    provides info about the configuration. This is used to identify different 
    simulations.
    
    Note that the simulations contain image stacks in the order (x,y,z), 
    while the output dataset contains the images as z,y,x.
    """
    # TODO: code doubling with load_vascu_synth_data
    with open(path.join(directory, dataset_specs), "r") as file:
        dataset_specs = json.loads(file.read())
    
    # get info about dataset:
    n_images = dataset_specs["images"]
    train_images =  dataset_specs["train_images"]
    n_train_images = len(train_images)
    test_images = dataset_specs["test_images"]
    n_test_images = len(test_images)    
    if n_train_images + n_test_images != n_images:
        warn("check train and test folds, because " +
             "not all images from dataset are used!")
        
    # TODO: infering these from each image individually would allow varying values    
    im_shape = dataset_specs["im_shape"]  # TODO: shape should be infered from data
    n_channels = dataset_specs["channels"]
    
    # Define which dimensions in the input corresponds to x,y,z
    # The spatial dimensions are arbitrary for synthetic data, but since 
    # sampling in z is usually lower in real data and psf is anisotropic in z, 
    # some care should be taken here to be consistent.
    input_z = 0
    input_y = 1
    #input_x = 0  # redundant
    
    train_shape = (n_train_images, im_shape[0], im_shape[1], 
                     im_shape[2], n_channels)
    test_shape = (n_test_images, im_shape[0], im_shape[1], 
                     im_shape[2], n_channels)

    # load data into 5d-numpy-arrays
    # X contains noisy images, Y ground-truth images (a.k.a. objects)
    X_train, Y_train = _create_numpy_dataset_split(directory, train_images, train_shape)
    X_test, Y_test = _create_numpy_dataset_split(directory, test_images, test_shape)
#    if i != n_test_images: 
#        warn("index has not reached n_test_images while filling X_test and "
#             "Y_test.")
 
    # swap if necessary so that dataset images have dimension in the order of
    # (depth, height, width) or (z, y, x) in my notation here.
    if input_z != 0:
        # swap 0 and 1 or 0 and 2
        # + 1 since in dataset the first dimension is n_images
        X_train = X_train.swapaxes(1, input_z+1)
        Y_train = Y_train.swapaxes(1, input_z+1)
        X_test = X_test.swapaxes(1, input_z+1)
        Y_test = Y_test.swapaxes(1, input_z+1)
    if input_y != 1 and input_z != 1:  # might already have swapped once
        # swap 1 and 2
        # + 1 since in dataset the first dimension is n_images
        X_train = X_train.swapaxes(2, input_y+1)
        Y_train = Y_train.swapaxes(2, input_y+1)
        X_test = X_test.swapaxes(2, input_y+1)
        Y_test = Y_test.swapaxes(2, input_y+1)
        
    return X_train, Y_train, X_test, Y_test

    
def load_vascu_synth_data(directory, dataset_specs="dataset_specs.json"):
    """
    Loads the images forming my modified version of the vascu-synth-dataset 
    from the appropriate directory and returns them as 5d-numpy-array with 
    dimensions (n_images, depth, height, width, channels).
    
    Args:
        directory (str): path to the dataset
        dataset_specs (str): path to json-file (relative from directory) 
                             specifying dataset structure
    
    Returns:
        X_train, Y_train, X_test, Y_test (numpy-arrays) with dimensions 
        (n_images, depth, height, width, channels) corresponding to images 
        and groundtruth objects.
    
    The vascu-synth dataset is available from http://vascusynth.cs.sfu.ca/).
    
    My modified version contains groundtruth images ("obj") and corresponding 
    simulated images ("im") as mat-files.
    The enclosing folder structure provides info about the configuration as 
    detailed in the README. This is used to identify different simulations.  
    The original group structure of the dataset is contained inside the 
    info-subfolders.
    
    Note that the simulations contain image stacks in the order (x,y,z), 
    while the output dataset contains the images as z,y,x.
    """
    with open(path.join(directory, dataset_specs), "r") as file:
        dataset_specs = json.loads(file.read())
    
    # get info about dataset:
    n_images = dataset_specs["images"]
    train_images =  dataset_specs["train_images"]
    n_train_images = len(train_images)
    test_images = dataset_specs["test_images"]
    n_test_images = len(test_images)    
    if n_train_images + n_test_images != n_images:
        warn("check train and test folds, because " +
             "not all images from dataset are used!")
        
    # TODO: infering these from each image individually would allow varying values    
    im_shape = dataset_specs["im_shape"]  # TODO: shape should be infered from data
    n_channels = dataset_specs["channels"]
    
    # Define which dimensions in the input corresponds to x,y,z
    # The spatial dimensions are arbitrary for synthetic data, but since 
    # sampling in z is usually lower in real data and psf is anisotropic in z, 
    # some care should be taken here to be consistent.
    input_z = 2
    input_y = 1
    #input_x = 0  # redundant
    
    train_shape = (n_train_images, im_shape[0], im_shape[1], 
                     im_shape[2], n_channels)
    test_shape = (n_test_images, im_shape[0], im_shape[1], 
                     im_shape[2], n_channels)

    # load data into 5d-numpy-arrays
    # X contains noisy images, Y ground-truth images (a.k.a. objects)
    X_train, Y_train = _create_numpy_dataset(directory, train_images, train_shape)
    X_test, Y_test = _create_numpy_dataset(directory, test_images, test_shape)
#    if i != n_test_images: 
#        warn("index has not reached n_test_images while filling X_test and "
#             "Y_test.")
 
    # swap if necessary so that dataset images have dimension in the order of
    # (depth, height, width) or (z, y, x) in my notation here.
    if input_z != 0:
        # swap 0 and 1 or 0 and 2
        # + 1 since in dataset the first dimension is n_images
        X_train = X_train.swapaxes(1, input_z+1)
        Y_train = Y_train.swapaxes(1, input_z+1)
        X_test = X_test.swapaxes(1, input_z+1)
        Y_test = Y_test.swapaxes(1, input_z+1)
    if input_y != 1 and input_z != 1:  # might already have swapped once
        # swap 1 and 2
        # + 1 since in dataset the first dimension is n_images
        X_train = X_train.swapaxes(2, input_y+1)
        Y_train = Y_train.swapaxes(2, input_y+1)
        X_test = X_test.swapaxes(2, input_y+1)
        Y_test = Y_test.swapaxes(2, input_y+1)
        
    return X_train, Y_train, X_test, Y_test


def _create_numpy_dataset(data_path, image_indices, data_shape):
    # X contains noisy images, Y ground-truth images (a.k.a. objects)    
    X = np.zeros(data_shape)  # , dtype=np.float32)
    Y = np.zeros(data_shape)  # , dtype=np.float32)
    for i, idx in enumerate(image_indices):
        data_name = 'image%d' % idx  # specific to structure of dataset
        im = load_mat_v7("im.mat", path.join(data_path, data_name))
        obj = load_mat_v7("obj.mat", path.join(data_path, data_name))
        # group structure is unwrapped into a linear list 
        # from 0 to n_images-1
        # print(i)
        X[i] = im[..., np.newaxis] # add empty channels dimension
        Y[i] = obj[..., np.newaxis] # add empty channels dimension
        print(i, "/", len(image_indices)-1)
    return X, Y

def _create_numpy_dataset_split(data_path, image_indices, data_shape):
    # X contains noisy images, Y ground-truth images (a.k.a. objects)
    dataset_shape = list(data_shape)
    dataset_shape[0] = 4*dataset_shape[0]
    dataset_shape = tuple(dataset_shape)
    
    X = np.zeros(dataset_shape)  # , dtype=np.float32)
    Y = np.zeros(dataset_shape)  # , dtype=np.float32)
    for i, idx in enumerate(image_indices):
        data_name = 'image%d' % idx  # specific to structure of dataset
        im1 = load_mat_v7("im1.mat", path.join(data_path, data_name))
        obj1 = load_mat_v7("obj1.mat", path.join(data_path, data_name))
        im2 = load_mat_v7("im2.mat", path.join(data_path, data_name))
        obj2 = load_mat_v7("obj2.mat", path.join(data_path, data_name))
        im3 = load_mat_v7("im3.mat", path.join(data_path, data_name))
        obj3 = load_mat_v7("obj3.mat", path.join(data_path, data_name))
        im4 = load_mat_v7("im4.mat", path.join(data_path, data_name))
        obj4 = load_mat_v7("obj4.mat", path.join(data_path, data_name))
        # group structure is unwrapped into a linear list 
        # from 0 to n_images-1
        # print(i)
        X[i]                   = im1[..., np.newaxis] # add empty channels dimension
        X[i + data_shape[0]]   = im2[..., np.newaxis]
        X[i + 2*data_shape[0]] = im3[..., np.newaxis]
        X[i + 3*data_shape[0]] = im4[..., np.newaxis]
        Y[i]                   = obj1[..., np.newaxis] # add empty channels dimension
        Y[i + data_shape[0]]   = obj2[..., np.newaxis]
        Y[i + 2*data_shape[0]] = obj3[..., np.newaxis]
        Y[i + 3*data_shape[0]] = obj4[..., np.newaxis]
        print(i, "/", len(image_indices)-1)
    return X, Y


def load_mat_v7(file_name, file_path="."):
    """
    Load a v7 (not v7.3!) mat-file generated with matlab's save-function like
    "save('data', 'data', '-v7')".
    
    Argss:
        file_name (str) should be in the form "name.mat"
        file_path (str) path to the file.  Default works on linux for cwd.
    
    Returns:
        data (numpy-array)
    
    I always use this convention for my mat-files.
    Might also work with older mat files.
    """
    # example: im = io.loadmat("im.mat")["im"] #[0][0][0] if saved from dip_image
    return io.loadmat(path.join(file_path, file_name))[file_name[0:-4]] 


def _define_train_test_fold(pairs_path, target_path, train_percentage=0.8, 
                            n_channels=1, im_shape=(400,100,100), 
                            file_name="dataset_specs.txt"):
    """
    provides a json-file to divide vascusynth-data into train and test folds.
    """
    n_images = len(next(os.walk(pairs_path))[1])  # count number of folders
    # TODO: infer im_shape and n_channels from dataset (if they are the same
    # for all)

    d = dict()
    d["images"] = n_images
    d["train_images"] = list(range(int(n_images*train_percentage)))
    d["test_images"] = list(range(int(n_images*train_percentage),n_images))
    d["channels"] = n_channels
    d["im_shape"] = im_shape
    
    # write to json
    with open(path.join(target_path, file_name), 'x') as file:
        file.write(json.dumps(d))


# %%
def main():    
    dataset_subpath = "small_test_noise/"
    simulation_subpath = "poisson/num_photons100000_bgr10_same/" + \
                         "na1.064_ri1.33_scaleXY61_scaleZ61/"
    train_percentage = 0.8
    im_shape = (400,100,100)  # need to change simulation subpath, when changing this
    
    # usually these do not need to be changed
    # assumes that all datasets in dataset_path have same n_images
    base_path = "/media/soenke/Data/Soenke/datasets/vascu_synth/"
    dataset_path = base_path + dataset_subpath
    pairs_path = dataset_path + "simulated_data_pairs/" + simulation_subpath
    target_path = pairs_path
    

    # if not exists ...
    if not path.isfile(path.join(target_path, "dataset_specs.json")):
        print("defining train/test fold")
        _define_train_test_fold(pairs_path, target_path, train_percentage,
                                n_channels=1, im_shape=im_shape,
                                file_name="dataset_specs.json")
    else:
        print("dataset specs already exist in target_path " + target_path +
              ". Will not generate new dataset specs. " +
              "Please resolve this manually.")

    # First solution if it is just a little too big: just fill train and
    # test separately

    # TODO: load bit by bit, if it's too big.
    print("loading data")
    X_train, Y_train, X_test, Y_test = load_vascu_synth_data(pairs_path)
    # use this instead for the old datasets seed2...5:
    # X_train, Y_train, X_test, Y_test = load_vascu_synth_data_split(
    #         dataset_path + simulation_subpath)

    # TODO: fill bit by bit, if it's too big 
    print("filling dataset")
    # generate real dataset
    print("---> deconv")
    create_datasets(X_train, Y_train, X_test, Y_test, target_path + "h5_deconv_dataset/")

    # generate identity-dataset
#    print("---> identity")
#    create_datasets(Y_train, Y_train, Y_test, Y_test,
#                      target_path + "identity")
    print("...done!")

if __name__ == '__main__':
    main()
