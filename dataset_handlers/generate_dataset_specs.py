#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 18:41:32 2018

@author: soenke
"""
import os
import json
from warnings import warn
import numpy as np
#from .dataset_utils import np_load

# write dataset specs, a json file that is used by list_dataset_handler

def write_dataset_specs(
        pairs_path, x_fname, y_fname, load_fn, train_fraction=0.8, 
        validation_fraction=0.0, dataset_id=None, x_format=None, y_format=None,
        cal_mean=True, cal_std=True, target_path=None, 
        out_name="dataset_specs.json"):
    """
    Assumes data is provided as folders named "image0", "image1", ... in
    pairs_path.
    Each folder should contain an input image with x_fname and a ground truth
    output image with y_fname.
    
    It will generate a json-file that defines which image is in train, 
    validation and test folds.  train_fraction and validation_fraction are 
    both with repsect to all images in the dataset.  The rest of the images 
    will be in "test_images". If n_images*train_fraction or 
    n_images*validation_fraction is not an integer, it will round down.   
    Note that this function does not shuffle images.  It simply
    puts the first number of images in train, the next ones in validation and
    the remaining in test.
    
    dataset_id, x_format and y_format may be provided as additional info on the
    dataset.  They are not required, but it is recommended to set them.
    
    - dataset_id is a string that can be used to identify the dataset.
    - x_format and y_format should be format strings such as "dhwc" (that is, 
      depth, height, width, channels) that define what each dimension stands 
      for in an x- or y-image.  
      Typical examples for 3D images:
          
      - "dhwc" (channels last), 
      - "cdhw" (channels first), 
      - "dhw" (no channel axis, eg. black/white image) 
      - "hwd" (no channel axis; z-axis (depth) is last)
    
    If cal_mean = True or cal_std = True, it will calculate dataset statistics
    of the training set for preprocessing.  Execution may take a little longer
    then.  Note that cal_std automatically also calculates mean
    """
    if train_fraction + validation_fraction > 1:
        raise ValueError("sum of train_fraction and validation_fraction " +
                         "cannot be > 1")
    # separate into training, validation and test set by indices.
    n_images = len(next(os.walk(pairs_path))[1])  # count number of folders
    train_images = list(range(int(n_images*train_fraction)))
    if not train_images:
        warn("train_images is empty.")
    validation_images = list(range(
            len(train_images), 
            len(train_images) + int(n_images*validation_fraction)))
    test_images = list(range(
            len(train_images) + len(validation_images),
            n_images))
    if not test_images:
        warn("test_images is empty.  Consider reducing train_fraction or " +
             "validation fraction.")
    
    # determine image properties.  They are assumed to be the same for 
    # the entire dataset.
    # Note that spatial shape (im_shape) must be the same for net x (input) 
    # and y (output).  The number of channels can be different.
    
    train_filelist = [os.path.join(pairs_path, "image" + str(i)) 
                      for i in train_images]
#    x_filetype = os.path.splitext(x_fname)[1][:-1]
#    y_filetype = os.path.splitext(y_fname)[1][:-1]
    
    # load_pair first pair from filelist
    np_x_tmp = load_fn(os.path.join(train_filelist[0], x_fname))
    np_y_tmp = load_fn(os.path.join(train_filelist[0], y_fname))
    x_shape = np_x_tmp.shape
    y_shape = np_y_tmp.shape
#    x_channels = np_x_tmp.shape[-1]
#    y_channels = np_y_tmp.shape[-1]

    # cal mean and std
    # This might have higher numerical accuracy, but is veery slow
    # and it only supports scalar mean and std, not feature-wise
#    if cal_mean or cal_std:
#        def xgen():
#            for file_path in filelist:
#                X = tb.load_mat_v7("im.mat", file_path)  # np-array
#                for pixel in X.flatten():
#                    yield pixel
#        if cal_mean:
#            X = xgen()
#            self.mean = statistics.mean(X)
#
#        if cal_std:
#            X = xgen()
#            self.std = statistics.stdev(X, mean=self.mean)
    
    # TODO: allow to choose whether to calculate mean across all pixels or
    #       keep a pixelwise mean.
    if cal_std: # automatically also calculates mean
        if not cal_mean:
            print("cal_std automatically also calculates mean.")
        print("calculating mean and std")
        means = list()
        sqr_means = list()
        for file_path in train_filelist:
            x = load_fn(os.path.join(file_path, x_fname)).astype(np.float64)
            sqr_means.append((x**2).mean())
            means.append(x.mean())
        mean = np.mean(means)
        std = np.sqrt(np.mean(sqr_means) - mean**2)
    elif cal_mean: # only
        print("calculating mean")
        means = list()
        for file_path in train_filelist:
            x = load_fn(os.path.join(file_path, x_fname)).astype(np.float64)
            means.append(x.mean())
        mean = np.mean(means)
        print("done")
        std = None
    else:
        mean = None
        std = None

    d = dict()
    d["n_images"] = n_images
    d["train_images"] = train_images
    d["validation_images"] = validation_images
    d["test_images"] = test_images
    d["x_name"] = x_fname
    d["y_name"] = y_fname
    d["x_shape"] = x_shape
    d["y_shape"] = y_shape
    d["x_format"] = x_format
    d["y_format"] = y_format
    d["dataset_id"] = dataset_id
    d["training_mean"] = mean
    d["training_std"] = std 
    
    # write to json
    if target_path is None:
        target_path = pairs_path
    try:
        with open(os.path.join(target_path, out_name), 'x') as file:
            file.write(json.dumps(d))
        print("have written dataset specifications to ", 
              os.path.join(target_path, out_name))
    except FileExistsError as e:
        print("dataset specs already exist in target_path " + target_path +
              ". Will not generate new dataset specs. " +
              "Please resolve this manually.")
        raise e
    return d


# %% example specific to my dataset
def main():
    # params
    num_photons=10000
    wl=1040
    train_fraction = 0.8
    validation_fraction = 0.05
    
    # usually these do not need to be changed for my datasets
    base_path = "/home/ziemer/datasets/vascu_synth/"
    pairs_subpath = "simulated_data_pairs/"
    dataset_subpath = "small/"
    bgr = int(num_photons//1000) or 1
    x_fname = "im.mat"
    y_fname = "obj.mat"

    # process params
    dataset_id = "poisson_n" + str(num_photons) + "_wl" + str(wl)
    noise_subpath = "poisson/num_photons"+str(num_photons)+"_bgr"+str(bgr)+"_same/"
    psf_subpath = "wl"+str(wl)+"_na1.064_ri1.33_scaleXY61_scaleZ61/"
    pairs_path = (base_path + dataset_subpath + pairs_subpath + 
                   noise_subpath + psf_subpath)
    target_path = pairs_path  # same as default

    write_dataset_specs(
            pairs_path, # path to folder containing folders "image0" etc.
            x_fname, # name of files in folders "image0" etc.
            y_fname,
            train_fraction=train_fraction, 
            validation_fraction=validation_fraction, 
            dataset_id=dataset_id,  # give dataset a meaningful name
            cal_mean=True,
            cal_std=True,
            target_path=target_path,
            out_name="dataset_specs.json")

if __name__ == '__main__':
    main()
