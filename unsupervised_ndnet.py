#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 13:00:57 2018

@author: Soenke
"""

import numpy as np
import tensorflow as tf
from warnings import warn

from ndnet import NDNet
from decorators import deprecated
import dataset_handlers as dh
import datetime

#%% unsupervised

class UnsupervisedNDNet(NDNet):
    " has extra args 'psf' and 'background' "
    def __init__(self, sess, psf, background, arch="unetv3", padding="same", 
             # net features
             force_pos=False, use_preprocessing=False,
             # training features
             use_batch_renorm=False, use_batch_norm=False, 
             # details
             last_layer_batch_norm=None, 
             data_format="channels_last",
             dataset_means=None, dataset_stds=None,
             comment=None):
        # all args except psf and background
        super().__init__(
                sess=sess, arch=arch, padding=padding,
                force_pos=force_pos, use_preprocessing=use_preprocessing, 
                use_batch_renorm=use_batch_renorm, 
                use_batch_norm=use_batch_norm, 
                last_layer_batch_norm=last_layer_batch_norm,
                data_format=data_format, dataset_means=dataset_means, 
                dataset_stds=dataset_stds, comment=comment)
        if data_format != "channels_last":
            raise NotImplementedError("need to fix dim_expansion of psf for data format other than channels_last")
        # psf lies in folder image_0 in dataset 
        # -> TODO: move psf to dataset handler? 
        self.psf = self.pretend_network_pass(tf.constant(
                np.expand_dims(np.expand_dims(psf,-1),0), name="psf"), mode="batch")  
        self.background = background  # TODO: should be > 1,  e.g. 1 or min(X)
        print("UNSUPERVISED LEARNING")
        print("... still experimental.")
    
    def train(self, training_data_path, n_epochs, batch_size, ckpt=None, 
              load_step=None, optimizer_fn="default", learning_rate_fn="default",
              loss_fn="l2loss", cut_loss_to_valid=False, weight_reg_str=None, 
              weight_reg_fn=None, data_reg_str=None, data_reg_fn=None,
              random_seed=None, dropout_rate=0.0, weight_init="default", 
              batch_renorm_scheme="default", dataset_handler="tfdata-list", 
              validate=False, comment=None):
        if batch_size > 1:
            raise NotImplementedError("convolution with psf has not been " +
                          "implemented for batch_size>1")
        return super().train(
                training_data_path=training_data_path, n_epochs=n_epochs, 
                batch_size=batch_size, ckpt=ckpt, load_step=load_step, 
                optimizer_fn=optimizer_fn, learning_rate_fn=learning_rate_fn, 
                loss_fn=loss_fn,  cut_loss_to_valid=cut_loss_to_valid, 
                weight_reg_str=weight_reg_str, weight_reg_fn=weight_reg_fn, 
                data_reg_str=data_reg_str, data_reg_fn=data_reg_fn,
                random_seed=random_seed, dropout_rate=dropout_rate, 
                weight_init=weight_init, batch_renorm_scheme=batch_renorm_scheme, 
                dataset_handler=dataset_handler, validate=validate, 
                comment=comment)
    
    def calculate_loss(self, x_batch, y_predicted_batch, 
                       loss_fn="l2loss", cut_loss_to_valid=False):
        """
        calculate loss by comparing blurred predicted y with x as in maximum 
        likelihood method.
        "poissonloss" is highly recommended for poisson noise
        """

        # This is extra to previous calculate_loss
        with tf.name_scope("fwd"):
            # convolve x_batch with psf
            
            # tensorflow's FFT does not accept non-complex values
            # tested direct convolution (tf.conv3d) but it is very slow
            # note ifftshift of psf above
            y_predicted_batch_cmplx = tf.squeeze(tf.complex(
                    y_predicted_batch, tf.zeros(self.psf.shape)),[0,-1])
            psf_cmplx = tf.squeeze(tf.complex(self.psf, tf.zeros(self.psf.shape)),[0,-1])
            # note that _ftconvolve takes complex tensors and returns a real one
            # TODO: is this correct?  is output real??
            # TODO: only implemented for batch_size==1
            # TODO: will not work for batch size >1
            y_predicted_batch_blur = tf.expand_dims(tf.expand_dims(_ftconvolve(y_predicted_batch_cmplx, psf_cmplx) + 
                                      self.background, 0),-1)
         
        warn("adding 1 to both to make use of poisson loss possibly possible")
        return super().calculate_loss(
                x_batch, y_predicted_batch_blur, loss_fn=loss_fn, 
                cut_loss_to_valid=cut_loss_to_valid)

    def _create_dataset(
            self, datadir, dataset_handler="tfdata-list",  mode="training"):
        # only change: create "UNSUPERVISEDListDatasetHandler
        
        """
        create dataset from datadir.  Mode can be "training", "validation" or
        "testing"
        """
        if mode not in ["training", "validation", "testing"]:
            raise ValueError("only modes ['training', 'validation', 'testing'] " +
                             "are allowed.")
        
        # TODO: mean of dataset must be a scalar
        if np.shape(self.dataset_means):
            raise RuntimeError("dataset_means is only implemented for scalars here.")
        if np.shape(self.dataset_stds):
            raise RuntimeError("dataset_stds is only implemented for scalars here.")   

        if dataset_handler == "tfdata-list":
            print("tfdata-list loads a list of image indices and provides " +
                  "methods to load images from the list in the tfdata-pipeline.")
            if mode == "training":
                field = "train_images"
            elif mode == "validation":
                field = "validation_images"
            elif mode == "testing":
                field = "test_images"
            dataset = dh.tfdata_dataset_handlers.UnsupervisedListDatasetHandler(
                    datadir, field=field, mean=self.dataset_means,
                    std=self.dataset_stds)
        else:
            raise ValueError(
                    "dataset_handler '" + str(dataset_handler) + 
                    "' is not supported for unsupervised training. " + 
                    "Choose one of [tfdata-list].")
        return dataset

def _ftconvolve(tensor1, tensor2):
    return tf.real(tf.ifft3d(tf.fft3d(tensor1) * tf.fft3d(tensor2)))

@deprecated("needs update")
def _unsupervised_train(
        training_data_path, psf, background, n_epochs=2, batch_size=2, ckpt=None, 
        load_step=None, optimizer_fn="default", learning_rate_fn=lambda lr: 1e-4,
        arch="unetv3", padding='same', force_pos=False, use_preprocessing=False, 
        use_batch_renorm=False, use_batch_norm=False, dropout_rate=0.0,
        loss_fn="l2loss", cut_loss_to_valid=False, weight_reg_str=None, 
        data_reg_str=None, random_seed=1, sess_config=None, validate=False,
        dataset_means=None, dataset_stds=None, comment=None):
    try:
        with tf.Session(config=sess_config) as sess: 
            deconv_net = UnsupervisedNDNet(
                    sess=sess, 
                    psf=psf,
                    background=background,
                    arch=arch, 
                    padding=padding,
                    force_pos=force_pos,
                    use_preprocessing=use_preprocessing,
                    use_batch_renorm=use_batch_renorm, 
                    use_batch_norm=use_batch_norm, 
                    last_layer_batch_norm=None, # True if any is used
                    data_format="channels_last",
                    dataset_means=dataset_means,
                    dataset_stds=dataset_stds)
            deconv_net.train(
                    training_data_path=training_data_path, 
                    n_epochs=n_epochs, 
                    batch_size=batch_size, 
                    ckpt=ckpt, 
                    load_step=load_step, 
                    dataset_handler="tfdata-list",
                    learning_rate_fn=learning_rate_fn,
                    optimizer_fn=optimizer_fn, 
                    loss_fn=loss_fn, 
                    cut_loss_to_valid=cut_loss_to_valid, 
                    weight_reg_str=weight_reg_str, 
                    weight_reg_fn="l2reg",
                    data_reg_fn="tv",
                    data_reg_str=data_reg_str,
                    random_seed=random_seed, 
                    weight_init="default", 
                    batch_renorm_scheme="default",
                    dropout_rate=dropout_rate,
                    validate=validate,
                    comment=comment)
        tf.reset_default_graph()
    except Exception as e:
        tf.reset_default_graph()
        raise e
    print("done")
    print(datetime.datetime.now().time())    