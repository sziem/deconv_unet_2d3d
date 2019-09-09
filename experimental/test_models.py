#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 00:15:21 2018

@author: soenke
"""

from deconv_net import _run_on_image, _test
from dataset_handlers.tfdata_dataset_handlers import (ListDatasetHandler, _dataset_mean_from_generator, _generator_from_h5)
import time
import tensorflow as tf
from toolbox import view3d as viewer
import numpy as np
#import os

# "test_images" : 170...199 except for w/o validation, those start at 160...199

def main():
    num_photons = 10000
    bgr = int(num_photons/1000)
    wl=1040
    image = 186 # from testset
    
    ckpt_num_photons = num_photons
    ckpt_bgr = bgr
    ckpt_wl=wl
    
    #path for data and ckpt
    noise_subpath = "poisson/num_photons"+str(num_photons)+"_bgr"+str(bgr)+"_same/"
    psf_subpath = "wl"+str(wl)+"_na1.064_ri1.33_scaleXY61_scaleZ61/"    

    #ckpt
    ckpt = ("/media/soenke/Data/Soenke/masterarbeit_final/combined/ckpts/small/poisson/" + 
            "num_photons"+str(ckpt_num_photons)+"_bgr"+str(ckpt_bgr)+"_same/"+
            "wl"+str(ckpt_wl)+"_na1.064_ri1.33_scaleXY61_scaleZ61/" + 
            "unetv3_depth3_valid0_nchs32_bn0_dr0_fp1_bs1_lr0.001_rseed1_avgp_adamnp_decay_poisson/" +
            "unetv3_depth3_valid0_nchs32_bn0_dr0_fp1_bs1_lr0.001_rseed1_avgp_adamnp_decay_poisson")
#    ckpt = ("/home/soenke/code/image_processing/unet_deconv/ckpts/small/poisson/" + 
#            "num_photons100000_bgr100_same/wl4160_na1.064_ri1.33_scaleXY61_scaleZ61/" + 
#            "unetv3_depth3_valid0_nchs32_bn0_dr0_bs1_lr3e-05_rseed2/" +
#            "unetv3_depth3_valid0_nchs32_bn0_dr0_bs1_lr3e-05_rseed2")
            #small/poisson/num_photons100000_bgr100_same/wl4160_na1.064_ri1.33_scaleXY61_scaleZ61/unetv3_depth3_valid0_nchs32_bn0_dr0_bs1_lr3e-05_rseed2" # "unetv3_depth3_valid0_nchs32_bn0_dr0_bs1_lr1e-05_rseed1_avgp_adamnp_test__" #"cluster/unetv3_depth3_valid0_nchs32_bn0_dr0_seed8_bs1_lr0.0003"
    load_step = 5440 #5760 #6400 #6080 #5440 #"previous" #43200
    
    # info prints
    loss_fn = "l2loss"  #"poissonloss", "huberloss"
    data_reg_str = None  #0.01
    cut_loss_to_valid=True  # does not matter for valid

    # network params
    padding='valid'
    use_batch_norm = False
    input_output_skip = False
    force_pos=True
    
    base_path = "/media/soenke/Data/Soenke/datasets/vascu_synth/"
    dataset_subpath = "small/"
    pairs_subpath = "simulated_data_pairs/"
    data_path = (base_path + dataset_subpath + pairs_subpath + 
                          noise_subpath + psf_subpath)
    testimage_path = (base_path + dataset_subpath + pairs_subpath + 
                      noise_subpath + psf_subpath + "image"+str(image)+"/")
    #ckpt_sub=dataset_subpath + noise_subpath + psf_subpath

    #n_images = ListDatasetHandler(training_data_path, cal_mean=False).n_images
    
    # setup preprocessing
    # TODO: change to doing this in dataset
    mean = ListDatasetHandler(data_path, cal_mean=True).mean
    std = 1  # -> takes long to calculate; not so important  # TODO: save in txt
    print(mean, std)

    #config=None    
    start = time.time()

    mean_loss = _test(data_path, 
                  #model
                  ckpt=ckpt, 
                  #ckpt_sub=dataset_subpath + noise_subpath + psf_subpath,
                  load_step=load_step, 
                  # network
                  padding=padding, 
                  input_output_skip=input_output_skip,
                  force_pos=force_pos,
                  use_batch_norm=use_batch_norm,
                  dataset_means=mean, 
                  dataset_stds=std,
                  # loss
                  loss_fn=loss_fn,
                  cut_loss_to_valid=cut_loss_to_valid,
                  data_reg_str=data_reg_str)

    x, y, yp = _run_on_image(testimage_path, 
                  #model
                  ckpt=ckpt, 
                  #ckpt_sub=dataset_subpath + noise_subpath + psf_subpath,
                  load_step=load_step, 
                  # network
                  padding=padding, 
                  input_output_skip=input_output_skip,
                  force_pos=force_pos,
                  use_batch_norm=use_batch_norm,
                  dataset_means=mean, 
                  dataset_stds=std,
                  # info prints
                  view_res=False,
                  cal_loss=True, 
                  loss_fn=loss_fn,
                  cut_loss_to_valid=cut_loss_to_valid,
                  data_reg_str=data_reg_str,
                  cal_time=False)
 
    # save
#    np.save("/media/soenke/Data/Soenke/masterarbeit_final/combined/arrays/img"+
#            "_"+str(num_photons)+"_"+str(wl)+"_poisson"+"_image"+str(image)+".npy",y)# +str(ckpt_num_photons)+"_"+str(ckpt_wl)+"_image"+str(image)+".npy",x)
#    np.save("/media/soenke/Data/Soenke/masterarbeit_final/combined/arrays/obj"+
#            "_"+str(image)+".npy",y)
    np.save("/media/soenke/Data/Soenke/masterarbeit_final/combined/arrays/est"+
            "_"+str(num_photons)+"_"+str(wl)+"_poisson"+"_image"+str(image)+".npy",yp)
    
    # view
    viewer.quick_max_projection_viewer(yp)
    viewer.quick_slice_viewer(yp)
    
    print(mean_loss)
    
    
    # TODO: change for unsupervised
    
#    _unsupervised_train(training_data_path,
#           _hrs_to_epochs(200), 
#           ckpt=ckpt,
#           load_step=load_step,
#           learning_rate_fn=learning_rate_fn,
#           optimizer_fn=optimizer_fn,
#           reg_str=reg_str, 
#           batch_size=batch_size,
#           padding=padding, 
#           loss_fn="poissonloss",
#           use_batch_norm=True if batch_size > 1 else False, 
#           random_seed=random_seed, 
#           dataset_means=mean, dataset_stds=std, # add to comment if used!
#           comment=('bs' + str(batch_size) +
#                    '_lr' + str(base_lr) + '_rseed' + str(random_seed)+ "_avgp_adamnp_decay_poisson_unsup"),
#           sess_config=config, 
#           ckpt_sub=dataset_subpath + noise_subpath + psf_subpath,
#           validate=True,
#           input_output_skip=False,
#           force_pos=True,
#           data_reg_str=data_reg_str)
    print("time: ", time.time()-start, "s")

if __name__ == '__main__':
    main()
