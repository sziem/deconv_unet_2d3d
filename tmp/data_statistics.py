#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:18:32 2018

@author: soenke
"""
import dataset_handlers as dh
import toolbox.toolbox as tb
import toolbox.view3d as viewer
import matplotlib.pyplot as plt

dataset_subpath = "seed8/"
base_path = "/media/soenke/Data/Soenke/datasets/vascu_synth/"
h5_subpath = "h5_deconv_dataset/"
pairs_subpath = "simulated_data_pairs/"
simulation_subpath = "poisson/num_photons10000_bgr10_same/" + \
                     "na1.064_ri1.33_scaleXY61_scaleZ244/"
training_data_path = (base_path + dataset_subpath + h5_subpath + 
                      simulation_subpath + "vascu_pairs_train.h5")
testimage_path = (base_path + dataset_subpath + pairs_subpath + 
                  simulation_subpath + "image10/")

mean = dh.tfdata_dataset_handlers.dataset_mean_from_generator(
            training_data_path, n_images=160)
#std = dh.tfdata_dataset_handlers.dataset_std_from_generator(
#            training_data_path, n_images=160)
std = 1
print(mean, std)

np_x = tb.load_mat_v7("obj.mat", testimage_path)
np_y = tb.load_mat_v7("im.mat", testimage_path)
tb.print_array_info(np_x, "obj", True)
tb.print_array_info(np_y, "im", True)

plt.figure()
bins = plt.hist(np_x.flatten(), bins=51)
bins = plt.hist(np_y.flatten(), bins=51)

plt.figure()
plt.plot(np_x.flatten(), np_y.flatten(), 'b.')
plt.show()

#viewer.quick_slice_viewer(np_y - np_x)
