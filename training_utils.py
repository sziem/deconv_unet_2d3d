#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 13:58:12 2018

@author: soenke
"""
#import tensorflow as tf
# for tensorflow 1.14 use this to avoid warnings:
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

def print_array_info(arr, name="array", minmax=True):
    """prints shape and dtype of numpy array"""
    print(name, "has shape", arr.shape)
    print(name, "has dtype", arr.dtype)
    if minmax == True:
        # TODO: is range a good name?  Nice: it fits the length of the others.
        print(name, "has range", [arr.min(), arr.max()])
              #"[{:0.1f},".format(arr.min()), "{:0.1f}]".format(arr.max()))


def _hrs_to_epochs(n_hrs):
    # estimate from 13.11.2018 (for normal unetv3)
    steps_per_hour = 1280 # roughly
    steps_per_epoch = 160 # dep on dataset
    epochs_per_hour = steps_per_hour/steps_per_epoch  # 8
    return int(n_hrs * epochs_per_hour)


def _learning_rate_with_decay(
        batch_size, batch_denom, num_images, boundary_epochs, decay_rates,
        base_lr=0.1, warmup=False):
    """
    Get a learning rate that decays step-wise as training progresses.
    Args:
        batch_size: the number of examples processed in each training batch.
        batch_denom: this value will be used to scale the base learning rate.
            `base_lr * batch size` is divided by this number, such that when
            batch_denom == batch_size, the initial learning rate will be base_lr.
            # --> why?
        num_images: total number of images that will be used for training.
        boundary_epochs: list of ints representing the epochs at which we
            decay the learning rate.
        decay_rates: list of floats representing the decay rates to be used
            for scaling the learning rate. It should have one more element
            than `boundary_epochs`, and all elements should have the same type.
        base_lr: Initial learning rate scaled based on batch_denom.
        warmup: Run a 5 epoch warmup to the initial lr.
    Returns:
        Returns a function that takes a single argument - the number of batches
        trained so far (global_step)- and returns the learning rate to be used
        for training the next batch.
    Note:
        taken from
        https://github.com/tensorflow/models/blob/master/official/resnet/resnet_run_loop.py
    """
    initial_learning_rate = base_lr * batch_size / batch_denom
    #batches_per_epoch = num_images / batch_size
    # change by soezie to adapt for case, where n_images/batch size does not fit.
    # in this case my tf-data input reduces batch size on last batch
    batches_per_epoch, last_batch_size = divmod(num_images, batch_size)
    if last_batch_size == 0:
        last_batch_size = batch_size  # this is actually not needed
    else:
        batches_per_epoch += 1
    
    # Reduce the learning rate at certain epochs.
    # CIFAR-10: divide by 10 at epoch 100, 150, and 200
    # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]
    
    def learning_rate_fn(global_step):
        with tf.name_scope("lr_decay"):
            """Builds scaled learning rate function with 5 epoch warm up."""
            lr = tf.train.piecewise_constant(global_step, boundaries, vals)
            if warmup:
                warmup_steps = int(batches_per_epoch * 5)
                warmup_lr = (
                    initial_learning_rate * tf.cast(global_step, tf.float32) / 
                    tf.cast(warmup_steps, tf.float32))
                return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
            return lr

    return learning_rate_fn


# %% weight initialization
    
# recommended for relu as in https://arxiv.org/pdf/1502.01852.pdf
he_init = tf.variance_scaling_initializer(scale=2.0, mode="fan_in", 
                                          distribution="truncated_normal")
# general
trunc_normal_init = tf.truncated_normal_initializer()


# %% helper for batch renorm

# note that this is quite slow for my data, which converges after around 10k 
# steps

# hacky scheme to roughly get batch_renorm_scheme 
# (from paper:  arXiv:1702.03275v2)
# "For Batch Renorm, we used rmax = 1, dmax = 0 (i.e. simply batchnorm) for the 
# first 5000 training steps, after which these were gradually relaxed to reach 
# rmax = 3 at 40k steps, and dmax = 5 at 25k steps. These final val- ues 
# resulted in clipping a small fraction of rs, and none of ds. However, at the 
# beginning of training, when the learning rate was larger, it proved important 
# to increase rmax slowly: otherwise, occasional large gradients were observed 
# to suddenly and severely increase the loss. To account for the fact that the 
# means and variances change as the model trains, we used relatively fast 
# updates to the moving statistics Î¼ and Ïƒ, with Î± = 0.01."
def default_batch_renorm_scheme(global_step):
    with tf.name_scope("batch_renorm_scheme"):
        rmax_start = 1
        dmax_start = 0
        
        rmax_end = 3
        dmax_end = 5
        
        # check wikipedia graph:
        # at t=-12 sigmoid is roughly zero and on plateau
        # t should be at -6 after about 5k steps to increase afterwards
        # then t should reach + 6 after 29k steps 
        # (somewhere between 25 and 40)
    
        # got these values from solving
        # t0 + 5 * incr = -6
        # t0 + 29 * incr = +6    
        #t = tf.Variable(-8.4995, name='t_batch_renorm_scheme', trainable=False)
        #incr = tf.constant(0.0005)
        #incr_t = tf.assign_add(t, incr)
        t = -8.5 + 0.0005*global_step
        
        # sigmoid goes from 0 to 1
        rmax = rmax_start + (rmax_end - rmax_start) * tf.sigmoid(t)
        dmax = dmax_start + (dmax_end - dmax_start) * tf.sigmoid(t)
        rmin = 1/rmax
        return rmin, rmax, dmax


# %% experimental models
        
# TODO: move this.  Does it work to put this in training utils?
# set model params
experimental_model_params = dict(
        # 2d
        _unetv3_2d=dict(
            network_depth=3,
            initial_channel_growth=32,
            channel_growth=2,
            conv_size=(3,3),  # y, x
            pool_size=(2,2),  # y, x
            input_output_skip=False,
            nonlinearity=tf.nn.relu),
        # change 1 thing
        _unetv3ioskip = dict(
            # set model params
            network_depth=3,
            initial_channel_growth=32,
            channel_growth=2,
            conv_size=(3,3,3),
            pool_size=(2,2,2),
            input_output_skip=True,
            nonlinearity=tf.nn.relu),
        _unetv3chgr3 = dict(
            network_depth=3,
            initial_channel_growth=32, # 64 not possible for 400x100x100 input
            channel_growth=3, #max 3 for 400x100x100 input
            conv_size=(3,3,3),
            pool_size=(2,2,2),
            input_output_skip=False,
            nonlinearity=tf.nn.relu), # must be function of a tensor or None
        _unetv3initch50 = dict(
            network_depth=3,
            initial_channel_growth=50, # max around 50 for 400x100x100 input; 
            channel_growth=2, #max 3 for 400x100x100 input and initial_ch=32
            conv_size=(3,3,3),
            pool_size=(2,2,2),
            input_output_skip=False,
            nonlinearity=tf.nn.relu), # must be function of a tensor or None
        _unetv3initch16 = dict(
            network_depth=3,
            initial_channel_growth=16, # max around 50 for 400x100x100 input; 
            channel_growth=2, #max 3 for 400x100x100 input and initial_ch=32
            conv_size=(3,3,3),
            pool_size=(2,2,2),
            input_output_skip=False,
            nonlinearity=tf.nn.relu), # must be function of a tensor or None
        # change 2 things
        _unetv3initch16chgr3 = dict(
            network_depth=3,
            initial_channel_growth=16, # max around 50 for 400x100x100 input; 
            channel_growth=3, #max 3 for 400x100x100 input
            conv_size=(3,3,3),
            pool_size=(2,2,2),
            input_output_skip=False,
            nonlinearity=tf.nn.relu), # must be function of a tensor or None
        _unetv3initch16chgr4 = dict(
            network_depth=3,
            initial_channel_growth=16, # max around 50 for 400x100x100 input; 
            channel_growth=4, #max 3 for 400x100x100 input
            conv_size=(3,3,3),
            pool_size=(2,2,2),
            input_output_skip=False,
            nonlinearity=tf.nn.relu), # must be function of a tensor or None
        _unetv3nd2chgr3 = dict(
            network_depth=2,
            initial_channel_growth=32, # max around 50 for 400x100x100 input; 
            channel_growth=3, #max 3 for 400x100x100 input and nd3
            conv_size=(3,3,3),
            pool_size=(2,2,2),
            input_output_skip=False,
            nonlinearity=tf.nn.relu), # must be function of a tensor or None
        _unetv3chgr3ioskip = dict(
            network_depth=3,
            initial_channel_growth=32,
            channel_growth=3,
            conv_size=(3,3,3),
            pool_size=(2,2,2),
            input_output_skip=True,
            nonlinearity=tf.nn.relu),
        _unetv3initch16convsize5 = dict(  # Problem: output becomes too small (cuts 88 pix)
            network_depth=3,
            initial_channel_growth=16,
            channel_growth=2,
            conv_size=(5,5,5),
            pool_size=(2,2,2),
            input_output_skip=False,
            nonlinearity=tf.nn.relu),
        _unetv3nd2convsize5 = dict(  # Problem: output becomes too small (cuts 88 pix)
            network_depth=2,
            initial_channel_growth=32,
            channel_growth=2,
            conv_size=(5,5,5),
            pool_size=(2,2,2),
            input_output_skip=False,
            nonlinearity=tf.nn.relu),
        # change 3 things
        _unetv3initch16convsize5ioskip = dict(  # Problem: output becomes too small (cuts 88 pix)
            network_depth=3,
            initial_channel_growth=16,
            channel_growth=2,
            conv_size=(5,5,5),
            pool_size=(2,2,2),
            input_output_skip=True,
            nonlinearity=tf.nn.relu),
        _unetv3nd2convsize5ioskip = dict(  # Problem: output becomes too small (cuts 88 pix)
            network_depth=2,
            initial_channel_growth=32,
            channel_growth=2,
            conv_size=(5,5,5),
            pool_size=(2,2,2),
            input_output_skip=True,
            nonlinearity=tf.nn.relu),
        _unetv3nd2convsize5chgr3 = dict(  # Problem: output becomes too small (cuts 88 pix)
            network_depth=2,
            initial_channel_growth=32,
            channel_growth=3,
            conv_size=(5,5,5),
            pool_size=(2,2,2),
            input_output_skip=False,
            nonlinearity=tf.nn.relu))