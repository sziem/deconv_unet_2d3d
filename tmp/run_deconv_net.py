#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 14:38:15 2018

@author: soenke
"""

import datetime
import tensorflow as tf
import os
import numpy as np

from deconv_net import DeconvNet3D, _hrs_to_epochs
import toolbox.view3d as viewer
import dataset_handlers as dh
import toolbox.toolbox as tb

def _train(training_data_path, n_epochs=2, ckpt="previous", lr=1e-4, 
           network_depth=3, padding='same', cut_loss_to_valid=False, 
           batch_size=2, n_channels_start=32, use_batch_norm=True, 
           use_dropout=False, load_step=None, random_seed=1, comment=None):
    try:
        optimizer=tf.train.AdamOptimizer(lr)
        with tf.Session() as sess:
            deconv_net = DeconvNet3D(sess, arch="unet", network_depth=network_depth,
                             padding=padding, n_channels_start=n_channels_start, 
                             use_batch_norm=use_batch_norm,
                             use_dropout=use_dropout)
            # max batch size that fits into memory is just 2!  -> not any more
            deconv_net.train(training_data_path,
                     n_epochs=n_epochs, batch_size=batch_size,
                     ckpt=ckpt, load_step=load_step,
                     dataset_handler="default",
                     optimizer=optimizer,
                     loss_fn="l2loss", cut_loss_to_valid=cut_loss_to_valid,
                     random_seed=random_seed, comment=comment)
        tf.reset_default_graph()
    except Exception as e:
        tf.reset_default_graph()
        raise e
    print("done")
    print(datetime.datetime.now().time())

def _train_deconv(n_epochs=2, ckpt="previous", lr=1e-4, network_depth=3, 
           padding='same', cut_loss_to_valid=False, batch_size=2, 
           n_channels_start=32, use_batch_norm=True, use_dropout=False, 
           load_step=None, comment=None):
    return _train(training_data_path="./testdata/" + 
            "poisson_num_photons10_bgr10_na1.2_ri1.33_scaleXY2000_scaleZ5000/" +
            "vascu_pairs_train.h5", n_epochs=n_epochs, ckpt=ckpt, lr=lr, 
            network_depth=network_depth, padding=padding, 
            cut_loss_to_valid=cut_loss_to_valid, batch_size=batch_size, 
            n_channels_start=n_channels_start, use_batch_norm=use_batch_norm, 
            use_dropout=use_dropout, load_step=load_step, comment=comment)

def _train_identity(n_epochs=2, ckpt="previous", lr=1e-4, network_depth=3, 
           padding='same', cut_loss_to_valid=False, batch_size=2, 
           n_channels_start=32, use_batch_norm=True, use_dropout=False, 
           load_step=None, comment=None):
    return _train(training_data_path="./testdata/identity/vascu_pairs_train.h5", 
            n_epochs=n_epochs, ckpt=ckpt, lr=lr, 
            network_depth=network_depth, padding=padding, 
            cut_loss_to_valid=cut_loss_to_valid, batch_size=batch_size, 
            n_channels_start=n_channels_start, use_batch_norm=use_batch_norm, 
            use_dropout=use_dropout, load_step=load_step, comment=comment)

def _test():
    pass

def _run_on_testimage(training_data_path, ckpt="previous", network_depth=3, 
                      padding="same", n_channels_start=32, use_batch_norm=True, 
                      load_step=None, cal_loss=False, loss_fn="l2loss", 
                      cut_loss_to_valid=True):
    """runs inference on random image from dataset"""
    
    # dropout is not applied in testing.  True and False would even lead
    # to same result due to the setting of is_training=False in inference
    use_dropout=False
    
    # load data:
    
    # unknown data
    # from scipy import io
    # imdir = "/home/soenke/Pictures/samples/3D/chromosome"
    # np_x = io.loadmat(os.path.join(imdir, "im.mat"))["im"][0][0][0]
    
    # data from training set using scipy.io.loadmat
    # from scipy import io
    # import toolbox.toolbox as tb
    # imdir = os.path.join("/home/soenke/Pictures/datasets/modified/",
    #                     "March_2013_VascuSynth_Dataset/simulated_data_pairs/",
    #                     "poisson/num_photons10_bgr10/",
    #                     "na1.2_ri1.33_scaleXY2000_scaleZ5000/Group1/data1"")
    # np_x = io.loadmat(os.path.join(imdir, "obj.mat"))["obj"][0][0][0]
    # tb.print_array_info(np_x, "x", True)
    
    # data directly from dataset
    dataset = dh.default_dataset_handler.DatasetHandler(training_data_path, 1)
    np_x, np_y = dataset.current_batch()
    np_x, np_y = np_x[0,:,:,:,0], np_y[0,:,:,:,0]
    tb.print_array_info(np_x, "x", True)
    tb.print_array_info(np_y, "y", True)
    
    try:
        with tf.Session() as sess:
            deconv_net = DeconvNet3D(sess, arch="unet", network_depth=network_depth,
                             padding=padding, n_channels_start=n_channels_start, 
                             use_batch_norm=use_batch_norm,
                             use_dropout=use_dropout)
            np_y_pred = deconv_net.run_on_image(np_x, ckpt, load_step)
            if cal_loss:
                print("loss: ", 
                      sess.run(deconv_net.calculate_loss(
                        tf.constant(np_y[np.newaxis,:,:,:,np.newaxis]), 
                        tf.constant(np_y_pred[np.newaxis,:,:,:,np.newaxis]), 
                        loss_fn=loss_fn, cut_loss_to_valid=cut_loss_to_valid)))
            if padding=="valid":
                # show only the part that corresponds to that output by the net
                np_x_crop = sess.run(deconv_net._crop_to_valid_network_output(
                 tf.constant(np.expand_dims(np.expand_dims(np_x,0),-1))))[0,:,:,:,0]
                np_y_crop = sess.run(deconv_net._crop_to_valid_network_output(
                 tf.constant(np.expand_dims(np.expand_dims(np_y,0),-1))))[0,:,:,:,0]
            else:
                # TODO: change name from "crop" to sth more sensible as it does not always crop.
                np_x_crop, np_y_crop = np_x, np_y
        tf.reset_default_graph()
    except Exception as e:
        tf.reset_default_graph()
        raise e
    
    #print(type(np_y_pred))
    # TODO: for valid: view/mark part that is considered in output
    viewer.quick_max_projection_viewer(np_x_crop)
    viewer.quick_max_projection_viewer(np_y_crop)
    viewer.quick_max_projection_viewer(np_y_pred)
    print("done")

def _run_on_testimage_deconv(ckpt="previous", network_depth=3, padding="same", 
                    n_channels_start=32, use_batch_norm=True, load_step=None, 
                    cal_loss=False, loss_fn="l2loss", cut_loss_to_valid=True):
    return _run_on_testimage("./testdata/" + 
            "poisson_num_photons10_bgr10_na1.2_ri1.33_scaleXY2000_scaleZ5000/" +
            "vascu_pairs_train.h5", ckpt=ckpt, network_depth=network_depth,
            padding=padding, n_channels_start=n_channels_start, 
            use_batch_norm=use_batch_norm, load_step=load_step, cal_loss=cal_loss, 
            loss_fn=loss_fn, cut_loss_to_valid=cut_loss_to_valid)

def _run_on_testimage_identity(ckpt="previous", network_depth=3, padding="same", 
                    n_channels_start=32, use_batch_norm=True, load_step=None, 
                    cal_loss=False, loss_fn="l2loss", cut_loss_to_valid=True):
    return _run_on_testimage("./testdata/identity/vascu_pairs_train.h5", 
            ckpt=ckpt, network_depth=network_depth, padding=padding, 
            n_channels_start=n_channels_start, use_batch_norm=use_batch_norm, 
            load_step=load_step, cal_loss=cal_loss, loss_fn=loss_fn, 
            cut_loss_to_valid=cut_loss_to_valid)

    
def main():    
    #_train_deconv(_hrs_to_epochs(200), None, 1e-3, 3, 'same', True, 2)
    _train_identity(_hrs_to_epochs(200), None, 1e-3, 3, 'valid', True, 2, comment="test")
    #_run_on_testimage_identity("previous", 3, "valid", cal_loss=True)    

if __name__ == '__main__':
    main()