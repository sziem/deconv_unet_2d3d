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


# %%

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

# for my own datasets
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

def _load_vascu_synth_train(directory, dataset_specs="dataset_specs.json"):
    
    # get info about dataset:
    with open(path.join(directory, dataset_specs), "r") as file:
        dataset_specs = json.loads(file.read())
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

    # load data into 5d-numpy-arrays
    # X contains noisy images, Y ground-truth images (a.k.a. objects)
    X_train, Y_train = _create_numpy_dataset(directory, train_images, train_shape)
 
    # swap if necessary so that dataset images have dimension in the order of
    # (depth, height, width) or (z, y, x) in my notation here.
    if input_z != 0:
        # swap 0 and 1 or 0 and 2
        # + 1 since in dataset the first dimension is n_images
        X_train = X_train.swapaxes(1, input_z+1)
        Y_train = Y_train.swapaxes(1, input_z+1)
    if input_y != 1 and input_z != 1:  # might already have swapped once
        # swap 1 and 2
        # + 1 since in dataset the first dimension is n_images
        X_train = X_train.swapaxes(2, input_y+1)
        Y_train = Y_train.swapaxes(2, input_y+1)
        
    return X_train, Y_train

def _load_vascu_synth_test(directory, dataset_specs="dataset_specs.json"):
    
    # get info about dataset:
    with open(path.join(directory, dataset_specs), "r") as file:
        dataset_specs = json.loads(file.read())
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
    
    test_shape = (n_train_images, im_shape[0], im_shape[1], 
                     im_shape[2], n_channels)

    # load data into 5d-numpy-arrays
    # X contains noisy images, Y ground-truth images (a.k.a. objects)
    X_test, Y_test = _create_numpy_dataset(directory, test_images, test_shape)
 
    # swap if necessary so that dataset images have dimension in the order of
    # (depth, height, width) or (z, y, x) in my notation here.
    if input_z != 0:
        # swap 0 and 1 or 0 and 2
        # + 1 since in dataset the first dimension is n_images
        X_test = X_test.swapaxes(1, input_z+1)
        Y_test = Y_test.swapaxes(1, input_z+1)
    if input_y != 1 and input_z != 1:  # might already have swapped once
        # swap 1 and 2
        # + 1 since in dataset the first dimension is n_images
        X_test = X_test.swapaxes(2, input_y+1)
        Y_test = Y_test.swapaxes(2, input_y+1)
        
    return X_test, Y_test


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
        X[i] = im[..., np.newaxis] # add empty channels dimension
        Y[i] = obj[..., np.newaxis] # add empty channels dimension
        #print(idx, "/", n_train_images)
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
    # example: im = io.loadmat("im.mat")["im"][0][0][0]
    return io.loadmat(path.join(file_path, file_name))[file_name[0:-4]][0][0][0]


def _define_train_test_fold(dataset_path, file_name="dataset_specs.txt"):
    """
    provides a json-file to divide vascusynth-data into train and test folds.
    """
    train_percentage = 0.8

    # TODO: infer these from dataset
    n_images = 200
    n_channels = 1
    im_shape = (100,100,100)

    d = dict()
    d["images"] = n_images
    d["train_images"] = list(range(int(n_images*train_percentage)))
    d["test_images"] = list(range(int(n_images*train_percentage),n_images))
    d["channels"] = n_channels
    d["im_shape"] = im_shape
    
    # write to json
    with open(path.join(dataset_path, file_name), 'x') as file:
        file.write(json.dumps(d))


# %%
# For old dataset
def load_vascu_synth_data_orig(directory, dataset_specs="dataset_specs.json"):
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
    
    n_groups = dataset_specs["n_groups"] # Do these correspond to same supply map?
    n_images_per_group = dataset_specs["n_images_per_group"] # Increasing index means larger bifurcation number
    n_images = n_groups * n_images_per_group
    n_channels = 1  # greyscale images
    
    train_groups =  dataset_specs["train_groups"]
    n_train_images = len(train_groups) * n_images_per_group
    test_groups = dataset_specs["test_groups"]
    n_test_images = len(test_groups) * n_images_per_group
    if n_train_images + n_test_images != n_images:
        warn("check train and test folds, because " +
             "not all images from dataset are used!")
    
    input_shape = (101,101,101)  # TODO: shape could be infered from data
    # Define which dimensions in the input corresponds to x,y,z
    # The spatial dimensions are arbitrary for synthetic data, but since 
    # sampling in z is usually lower in real data and psf is anisotropic in z, 
    # some care should be taken here to be consistent.
    input_z = 2
    input_y = 1
    #input_x = 0  # not needed
    
    train_shape = (n_train_images, input_shape[0], input_shape[1], 
                     input_shape[2], n_channels)
    test_shape = (n_test_images, input_shape[0], input_shape[1], 
                     input_shape[2], n_channels)
#    entire_shape = (n_images, input_shape[0], input_shape[1], 
#                     input_shape[2], n_channels)
    # X contains noisy images, Y ground-truth images (a.k.a. objects)
    X_train = np.zeros(train_shape)
    Y_train = np.zeros(train_shape)
    X_test = np.zeros(test_shape)
    Y_test = np.zeros(test_shape)

    # the psf is not needed.  Note that it is the same for all data-pairs.
    # psf = load_mat_v7(path.join(directory, "Group1", "data1", "psf.mat"))

    # load data into 5d-numpy-array
    idx = 0
    for i in train_groups:
        group_name = 'Group%d' % i # specific to structure of dataset
        for j in range(n_images_per_group):
            data_name = 'data%d' % (j+1)  # specific to structure of dataset
            im = load_mat_v7("im.mat",
                             path.join(directory, group_name, data_name))
            obj = load_mat_v7("obj.mat",
                              path.join(directory, group_name, data_name))
            # group structure is unwrapped into a linear list 
            # from 0 to n_images-1
            X_train[idx] = im[..., np.newaxis] # add empty channels dimension
            Y_train[idx] = obj[..., np.newaxis] # add empty channels dimension
            idx += 1
            #print(idx, "/", n_train_images)
    if idx != n_train_images: 
        warn("index has not reached n_train_images while filling X_train and "
             "Y_train.")

    #  TODO: code-doubling here
    idx = 0
    for i in test_groups:
        group_name = 'Group%d' % i # specific to structure of dataset
        for j in range(n_images_per_group):
            data_name = 'data%d' % (j+1)  # specific to structure of dataset
            im = load_mat_v7("im.mat",
                             path.join(directory, group_name, data_name))
            obj = load_mat_v7("obj.mat",
                              path.join(directory, group_name, data_name))
            # group structure is unwrapped into a linear list 
            # from 0 to n_images-1
            X_test[idx] = im[..., np.newaxis] # add empty channels dimension
            Y_test[idx] = obj[..., np.newaxis] # add empty channels dimension
            idx += 1
    if idx != n_test_images: 
        warn("index has not reached n_test_images while filling X_test and "
             "Y_test.")
 
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

def _define_train_test_fold_orig(dataset_path, file_name="dataset_specs.txt"):
    """
    provides a json-file to divide vascusynth-data into train and test folds
    by groups.
    """
    n_groups = 10
    groups = list(range(1, n_groups + 1))
    
    n_images_per_group = 12
    # images = list(range(1, n_images_per_group + 1))

    d = dict()
    d["n_groups"] = n_groups
    d["n_images_per_group"] = n_images_per_group
    d["train_groups"] = groups[0:8]  # 1...8
    d["test_groups"] = groups[8:n_groups]  # 9...10
    
    # provide other info such as image_shape
    # TODO: make sure to check that all images / groups are used
    
    # write to txt
    with open(path.join(dataset_path, file_name), 'x') as file:
        file.write(json.dumps(d))

#def _preprocessing_data(np_X, np_Y, subtract_mean, normalize):
#    """
#    perform data preprocessing.  
#    I don't think this should be done here though, but rather when 
#    giving input to CNN.
#    """  
#    mean = np.mean(np_X)
#    std = np.std(np_X)
#    # These are needed for comparing X and Y.
#    # In loss layer, mean subtraction and normalization need to be undone. 
#    #    np.save(os.path.join(target_path, "mean.npy"), mean)
#    #    np.save(os.path.join(target_path, "std.npy"), std)  
#    if subtract_mean:
#        np_X -= mean
#    if normalize:
#        np_X /= std
#    return np_X, mean, std


# %%
def main():
    dataset_path = "/media/soenke/Data/Soenke/datasets/vascu_synth/dataset1/" + \
                   "simulated_data_pairs/"
    target_path = "/media/soenke/Data/Soenke/datasets/vascu_synth/dataset1/" + \
                  "h5_deconv_dataset/"
    simulation_subpath = "poisson/num_photons10000_bgr10_same/" + \
                         "na1.064_ri1.33_scaleXY61_scaleZ244/"

    # if not exists ...
    if not path.isfile(path.join(dataset_path, simulation_subpath, 
                       "dataset_specs.json")):
        print("defining train/test fold")
        _define_train_test_fold(path.join(dataset_path, simulation_subpath), 
                                file_name="dataset_specs.json")

    # First solution if it is just a little too big: just fill train and
    # test separately


    # TODO: load bit by bit, if it's too big.
    print("loading data")
    X_train, Y_train, X_test, Y_test = load_vascu_synth_data(
            dataset_path + simulation_subpath)

    # TODO: fill bit by bit, if it's too big 
    print("filling dataset")
    # generate real dataset
    print("---> deconv")
    create_datasets(X_train, Y_train, X_test, Y_test,
                      target_path + simulation_subpath)

    # generate identity-dataset
    print("---> identity")
    create_datasets(Y_train, Y_train, Y_test, Y_test,
                      target_path + "identity")
    print("...done!")

if __name__ == '__main__':
    main()
