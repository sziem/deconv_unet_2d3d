#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:17:11 2018

@author: soenke
"""

# this is from https://github.com/abseil/abseil-py/issues/102
# it is apparently fixed in nightly tf-builds
try:
    import absl.logging
    absl.logging._warn_preinit_stderr = False
except Exception:
    print("tf 1.14.0 will print abseil.logging warning.")
    pass

import numpy as np
#import tensorflow as tf
import os
#import json
import csv
from warnings import warn
import datetime
from math import ceil
from itertools import permutations

# for tensorflow 1.14 use this to avoid some warnings:
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

# own modules
import dataset_handlers as dh
import loss_functions as lf
import network_architectures as na
import tools.training_utils as training_utils
from tools.training_utils import experimental_model_params
#import tools.decorators as decorators

# NOTE:
# if you get an error with mkl while running the test on cpu, 
# switch to the eigen builds of tensorflow

#IDEAS:
## build inference during init or using a build_network method
# --> would probably be cleaner

## configure session this way
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# config = tf.ConfigProto(gpu_options=gpu_options)
# sess = tf.Session(config)  # or similar
# /// or sth like
# session_config.gpu_options.visible_device_list= '1' #only see the gpu 1
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config, ...)
# config=tf.ConfigProto(log_device_placement=True)
# --> currently needs all memory anyways...

## network architecture:
# allow circular/reflection padding!
# change conv size in activ3 in unet3d.build_final_blocks to (1,1,1)
# --> test other settings first to see if this is necessary
# use fully connected layer in the end (?)

## adaption to input:
# allow different network_depths in different directions
# --> 

## pre- and post-processing:
# should I omit post-processing?
# should I scale output from postprocessing to 0..255 (?)
# --> experiment

## profile where work is done in net
# https://www.tensorflow.org/guide/graph_viz
# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# run_metadata = tf.RunMetadata()        
#    summary, _ = sess.run([merged, train_step],
#                           feed_dict=feed_dict(True),
#                           options=run_options,
#                           run_metadata=run_metadata)
#    train_writer.add_run_metadata(run_metadata, 'step%d' % i)
#    train_writer.add_summary(summary, i)
#    print('Adding run metadata for', i)
# --> interesting, but not so important
#
# histogram of activations
#    https://www.tensorflow.org/guide/tensorboard_histograms
# --> would be nice
#
# debugging
#    https://www.tensorflow.org/guide/debugger
# --> if necessary
#
# possibly omit _parse_load_step in favor of using sth like this:
# https://stackoverflow.com/questions/45077445/ ...
# ... how-to-use-method-recover-last-checkpoints-of-tf-train-saver

# max number of parameters is about 10 million, but it depends
# at which sampling they occur 

## use explicit graph 
# self.graph = tf.graph()
# with tf.Graph().as_default() as g (but graph is included in sess right?)
# --> need to learn more about tf internals

## specify data_format differently
# more explicit 2D/3D would be to specify ndhwc etc.

# posibly docment attributes here
class NDNet(object):
    def __init__(
            self, sess, 
            arch="unetv3", padding="same", net_channel_growth=1,
            force_pos=False, normalize_input=False, 
            use_batch_renorm=False, use_batch_norm=False, 
            last_layer_batch_norm=None, 
            custom_preprocessing_fn=None, custom_postprocessing_fn=None,
            dataset_means=None, dataset_stds=None, 
            data_format="channels_last", comment=None):
        """
        NDNet provides a framework for training, running and testing Neural
        Networks for image-to-image translation (any mapping from image scale
        to image scale such as deconvolution, denoising or semantic 
        segmentation).  

        Training and testing data consists of pairs of images (x,y), 
        where x and y must have the same spatial shape. 
        
        Both (2d+channels) and (3d+channels) images are supported.  The 
        number of input and output channels can be different (e.g. color input
        and black and white output).
        
        The dimensionality is determined by the network architecture.  If the
        CNN uses 3d convolution kernels, then the input image must be 3d.  
        EXPERIMENTAL: The number of input and output channels can be tuned 
        using the net_channel_growth parameter.  This can be a fraction 
        (e.g. 3 input channels and 1 output channel -> 1/3).
        
        This class manages and standardizes all tasks that are common to 
        training, running and testing such as:
          - managing tf-sessions (note: might be removed from in the future)
          - data loading
          - pre- and postprocessing
          - running the network for a given architecture
          - calculating loss
          - saving and loading checkpoints
        
        Args:
            sess: a tf-session
                .
            # net features
            
            arch (str) : defines network-architecture.  Currently only "unetv3" 
                and unetv3_small are officially supported.  Both are 3d models.
                There are also a number of experimental models 
                (such as "_unetv3_2d") in training_utils.  
                Note that the dimensionality of the model is implicitly 
                encoded in the network architecture.  If the model uses 3d
                convolution kernels, then the in- and output images must be 3d.
                Default: "unetv3" (a 3d model)
            padding (str) : padding of all convolutions.  Can be "same" or 
                "valid".  
                Default: "same"
            net_channel_growth (float) : EXPERIMENTAL: change this, if you 
                want to have a different number of in- and output channels.
                In initial tests, it was possible to set this even to float 
                values such as 2/3 (from 3 channel input to 2 channel output).
                But I am actually surprised that worked, so use this with care.
                Default: 1
            force_pos (bool) : add positivity constraint by squaring 
                network output.  Default: False
            normalize_input (bool) : Do preprocessing using mean subtraction 
                with dataset_means and std normalization using dataset_stds.
                Default: False
            use_batch_renorm (bool) : Add batch normalization layers and 
                use batch renormalization scheme as suggested in 
                arXiv:1702.03275v2.  
                Default: False
            use_batch_norm (bool) : Add batch normalization layers and perform
                standard batch normalization.  
                Default: False
            
            # details
            last_layer_batch_norm (None or bool) : perform batch_nom also in 
                last layer.  None means True if use_batch_(re)normalization
                else False.  Default: None
            custom_preprocessing_fn (None or function) : a function of a 
                single image from the dataset.  It must have one argument 
                (the input tensor) and return one output tensor
                Default: None
            custom_postprocessing_fn (None or function) : a function of a 
                single image from the dataset.  It must have one argument 
                (the input tensor) and return one output tensor
                Default: None
            dataset_means (None or float) : can be used for preprocessing. 
                Default: None
            dataset_stds (None or float) : can be used for preprocessing.
                Default: None

            data_format (str) : data format of arrays in tf-graph.  Used by
                tf.layers module.  Can be "channels_last" or "channels_first"
            comment (None or str) : will be added to the model_id, which is
                used to identify checkpoints
                Default: None
                
        Note:
            Preprocessing is applied in the order:: 
                force_pos_op -> custom_preprocessing -> normalize_input
            Postprocessing undoes this in the order::
                undo_normalization -> custom_postprocessing -> force_pos_op
        """
        # Note:
        # The network operates on minibatches of data that have shape 
        # (N, (D,) H, W, C), i.e. consisting of N images, each with (depth D,) 
        # height H and width W and with C input channels.
        self.sess = sess
        self.arch = arch # store more info in this
        self.force_pos = force_pos
        # it could be useful to have self.data_format, because in principle
        # pre- and postprocessing could have different data_format than
        # the model.  But for now this will not be allowed.
        # self.data_format = data_format
        
        self.normalize_input = normalize_input 
        if self.normalize_input:            
            if dataset_means is None and dataset_stds is None:
                raise ValueError(
                        "normalize_input is True, but neither dataset_means " +
                        "nor dataset_stds are provided.")
            elif dataset_means is None:
                print("Only dataset_stds was given.  Setting dataset_means " +
                      "to default value 0.")
                dataset_means = 0
            elif dataset_stds is None:
                print("Only dataset_means was given.  Setting dataset_stds " +
                      "to default value 1.")
                dataset_stds = 1
        if dataset_means == 0 and dataset_stds == 1:
            print("skipping preprocessing, because dataset_means is " +
                  "already 0 and dataset_stds is already 1.")
            self.normalize_input = False
        self.dataset_means = dataset_means
        self.dataset_stds = dataset_stds
        # TODO (EXPERIMENTAL -> TEST)
        # TODO: possibly do shape corrections before applying 
        #       custom_postprocessing_fn
        # TODO: 
        self.custom_preprocessing_fn = custom_preprocessing_fn
        self.custom_postprocessing_fn = custom_postprocessing_fn
        
        # define model
        if arch == "unetv3":
            network_depth=3
            initial_channel_growth=32
            channel_growth=2
            conv_size=(3,3,3)
            pool_size=(2,2,2)
            input_output_skip=False
            nonlinearity=tf.nn.relu # must be function of a tensor or None
        elif arch == "unetv3_small":
            network_depth=2
            initial_channel_growth=2
            channel_growth=2
            conv_size=(3,3,3)
            pool_size=(2,2,2)
            input_output_skip=False
            nonlinearity=tf.nn.relu # must be function of a tensor or None
        elif arch in experimental_model_params.keys():
            d = experimental_model_params[arch]
            network_depth = d["network_depth"]
            initial_channel_growth = d["initial_channel_growth"]
            channel_growth = d["channel_growth"]
            conv_size = d["conv_size"]
            pool_size = d["pool_size"]
            input_output_skip = d["input_output_skip"]
            nonlinearity = d["nonlinearity"]
        else:
            raise ValueError(
                    "Unsupported arch '" + arch + "'. " + 
                    "Currently the models 'unetv3' and 'unetv3_small' are " +
                    "officially supported.  Experimental models are " + 
                    str(list(experimental_model_params.keys())) + ".")
        self.model = na.unet.Unet_v3(
                padding=padding, 
                nonlinearity=nonlinearity, 
                network_depth=network_depth,
                net_channel_growth=net_channel_growth,  # experimental
                initial_channel_growth=initial_channel_growth, 
                channel_growth=channel_growth,
                conv_size=conv_size, 
                pool_size=pool_size,
                use_batch_renorm=use_batch_renorm, 
                use_batch_norm=use_batch_norm,
                last_layer_batch_norm=last_layer_batch_norm,
                data_format=data_format,
                input_output_skip=input_output_skip)
        
        # saving checkpoints in 
        # self.modeldir/model_id/dataset_id/run_id/run_0/"ckpts"/run_0".ckpt")
        # and logs in 
        # self.modeldir/model_id/dataset_id/run_id/run_0/"logs"/xxx".logs")
        # ---> just delete the folder containing model to delete both
        self.model_id = self._set_model_id(comment=comment)
        # TODO: possible os.path.abspath("./models") to fix windows-problem
        self.modeldir = "models"


    ## High level control
    def train(self, training_dataset_handler, n_epochs, batch_size, 
              optimizer_fn=lambda lr : tf.train.AdamOptimizer(lr), 
              learning_rate_fn=lambda training_step: 1e-3,
              loss_fn=tf.losses.mean_squared_error, cut_loss_to_valid=False, 
              weight_reg_str=None, weight_reg_fn=None, data_reg_str=None, 
              data_reg_fn=None,
              ckpt=None, load_step=None, random_seed=None, 
              weight_init=None, 
              batch_renorm_fn=None, dropout_rate=0.0, 
              validate=False, validation_dataset_handler=None,
              summary_interval=None, save_interval=None,
              comment=None):
        """
        Configure network for training,
        load training data 
        and run optimizer loop on vars to minimize loss.
        
        Args:            
            training_dataset_handler (dataset_handler) : dataset_handler from 
                dataset_handlers.tf_data_dataset_handlers.  These provide a thin
                layer around tf.data datasets.  If you have an existing tf.data
                dataset, using BaseDatasetHandler is sufficient.  The module also
                provides Handlers that can be initialized from numpy-arrays
                or lists of files.
            n_epochs (int) : number of training epochs.  An epoch is completed, 
                when the network has seen all images once.
            batch_size (int) : training batch size.
                .
            # optimization params
            
            optimizer_fn (function) : optimizer_fn must be a function 
                taking one parameter (learning rate) and return an optimizer 
                operation.  An optimizer can be transformed to an optimizer_fn 
                as simple as:
                    optimizer_fn = lambda lr: tf.train.AdamOptimizer(lr)
                This is also the default.
            learning_rate_fn (function) : learning_rate_fn must be a function 
                taking one parameter (global_step) and return a 
                float/tf.constant/tf.Variable. The function makes it easier to 
                define learning rate decay. 
                Default: A constant learning rate of 1e-3:
                    learning_rate_fn = lambda training_step: 1e-3
            
            # loss function
            
            loss_fn (function) : Loss_fn must be a function taking two parameters
                (labels, predictions) and return a loss-tensor.  Common losses can
                be found in tf.losses and in this repo's loss_functions (lf)
                Default: tf.losses.mean_squared_error
            cut_loss_to_valid (bool) :  This can be used for 'same' padding to
                calculate the loss only on that part of the image that is not
                impacted by padding.  This has no impact, if padding='valid'
                Default: False
            weight_reg_str (float) : float to scale the strength of weight 
                regularization (similar to weight decay)
            weight_reg_fn (function) : a function of a single weight, typically
                the square or absolute value.  All values of reg_fn(weight) are 
                added and multiplied by weight_reg_str and then added to the loss.
            data_reg_str (float) : float to scale the strength of data 
                regularization.
            data_reg_fn (function) : a function of a single image returning a loss
                tensor.  Its value is multiplied by data_reg_str and then added
                to the loss.  NOTE that this currently only works correctly for 
                a batch size of 1.
            
            # model loading or init
            
            ckpt (None or str) : path to checkpoint that will be loaded to 
                initialize training.  load_step should not be included in ckpt, 
                but should be provided separately.  If None, network will be 
                randomly initialized depending on weight_init and random_seed
            load_step (None, "previous" or int) : step that will be loaded from ckpt.
                "previous" is converted to the last ckpt that was written
                to the folder containing the ckpt.
                Cannot be None, if ckpt is not None.
            random_seed (None or int) : The random seed is passed to all random
                number generators, e.g. to shuffling of dataset and initialization 
                of weights.  If left as "None", the shuffling or initialization
                cannot be deterministically repeated.
                NOTE: This should be changed when loading from ckpt.  Otherwise
                the exact same sequence will be returned again.
            weight_init (None or initializer) : if no ckpt is provided, an 
                initializer (eg. from tf.initializers or from this repo's 
                training_utils must be provided. Recommendation for ReLU activation:
                training_utils.he_init
            
            # training specific features
    
            batch_renorm_fn (None or function) : batch_renorm_fn must be a function 
                taking one parameter (global_step) and return rmin, rmax, dmax 
                values for clipping as described in the batch renormalization paper.  
                See default_batch_renorm_scheme as an example, which imitates the
                suggested scheme from the paper.  If self.use_batch_renorm is 
                False, this has no impact.
                NOTE: I do not recommend using batch_renorm any more and this
                may disappear in the future.
            dropout_rate (float) : fraction of the activations that are dropped 
                out.
            validate (bool) : decides whether or not to put mean loss on 
                validation set in summary.  If True, validation_dataset_handler
                must also be provided.
            validation_dataset_handler (None or dataset_handler) : See 
                training_dataset_handler for details about dataset_handler.  The
                loss for every image in the validation_dataset will be calculated
                sequentially and then the mean is calculated.
                NOTE that this will slow down training if done frequently and on a
                large set.
            summary_interval (int or None) : number of steps after which a log
                is written.  Default (None): 2 logs per epoch
            save_interval (int or None) : number of steps after which a ckpt is 
                written.  Default (None): save ckpt after every 2 epochs
            comment (None or str) : comment can be provided to modify run_id to 
                label runs
            
        Returns:
            None.  But updates variables of model and saves ckpts.
            
        Raises:
            TODO
        """
        (dataset, training, total_loss, saver, new_ckpt, writer, summary, 
         validation_dataset, val_loss_single, val_loss_ph) = self._train_init(
                training_dataset_handler, n_epochs, batch_size, 
                # model loading or init
                ckpt=ckpt, load_step=load_step, random_seed=random_seed, 
                weight_init=weight_init, 
                # training specific args
                loss_fn=loss_fn, cut_loss_to_valid=cut_loss_to_valid, 
                weight_reg_str=weight_reg_str, weight_reg_fn=weight_reg_fn, 
                data_reg_str=data_reg_str, data_reg_fn=data_reg_fn, 
                batch_renorm_fn=batch_renorm_fn, 
                dropout_rate=dropout_rate, validate=validate, 
                validation_dataset_handler=validation_dataset_handler,
                optimizer_fn=optimizer_fn, learning_rate_fn=learning_rate_fn,
                # added to run_id
                comment=comment)
        self._train_loop(
                n_epochs, batch_size, dataset, total_loss, training, saver, 
                new_ckpt, writer, summary, validation_dataset, val_loss_single, 
                val_loss_ph, summary_interval=summary_interval, 
                save_interval=save_interval)
    
    # TODO: make trainer object
    def _train_init(
            self, training_dataset_handler, n_epochs, batch_size, 
            # model loading or init
            ckpt=None, load_step=None, random_seed=None, 
            weight_init=None, 
            # training specific args
            loss_fn=tf.losses.mean_squared_error, cut_loss_to_valid=False, 
            weight_reg_str=None, weight_reg_fn=None, data_reg_str=None, 
            data_reg_fn=None, batch_renorm_fn=None, dropout_rate=0.0,
            validate=False, validation_dataset_handler=None,
            optimizer_fn=lambda lr : tf.train.AdamOptimizer(lr), 
            learning_rate_fn=lambda training_step: 1e-3,
            # appended to run_id
            comment=None):
        # TODO: kwarg: add summary (and which kinds)
        is_training = True
        tf.set_random_seed(random_seed)  # for tf-random generators in the graph   

        # get infos from ckpt
        load_step = _parse_load_step(ckpt, load_step)
        if ckpt:
            if weight_init:
                warn("weight_init is " + str(weight_init) + ". It will be " +
                     "ignored.  Weights are not re-initialized, when loading " +
                     "from ckpt.")
            weight_init=None # already init'ed
            # TODO training will restart with same sequence if initialized
            # with same random seed.  Possible solutions that are not yet
            # implemented:
            # -> save random seed in ckpt/model id or use global_step 
            # -> or make random_seed placeholder/Variable
            # -> or better: save iterator state in ckpt!
            # -> just issue a warning
        else:
            ckpt_dataset_id = ""
            ckpt_run_id = ""

        # define basic run parameters     
        self.model.extra_training_parameters(
                weight_init=weight_init,  # do not change in case of relu
                batch_renorm_fn=batch_renorm_fn,
                dropout_rate=dropout_rate) # often not used

        with tf.name_scope("training_input_and_preprocessing"):
            dataset = self.TrainingDatasetAndPreprocess(
                    training_dataset_handler=training_dataset_handler, 
                    batch_size=batch_size, n_epochs=n_epochs, 
                    random_seed=random_seed)
            x_batch, y_batch = dataset.next_batch()
        input_shape = dataset.x_shape
        print("input_shape:", input_shape)
        
        with tf.variable_scope("model"):
            # x_batch is already preprocessed by dataset_handler
            y_predicted_batch = self.model.inference(x_batch, is_training)
        y_predicted_batch = self.postprocess(y_predicted_batch, input_shape)
        print("output_shape:", y_predicted_batch.shape)
        
        with tf.name_scope("losses"):
            data_loss, weight_reg_loss, data_reg_loss= self.calculate_losses(
                y_batch, y_predicted_batch, loss_fn=loss_fn, 
                cut_loss_to_valid=cut_loss_to_valid, 
                weight_reg_str=weight_reg_str, weight_reg_fn=weight_reg_fn, 
                data_reg_str=data_reg_str, data_reg_fn=data_reg_fn)
            total_loss = data_loss + weight_reg_loss + data_reg_loss

        if validate:
            with tf.name_scope("validation_input_and_preprocessing"):
                if validation_dataset_handler is None:
                    raise ValueError(
                            "validate is True, but " + 
                            "validation_dataset_handler is None.")
                validation_dataset = self.ValidationDatasetAndPreprocess(
                    validation_dataset_handler=validation_dataset_handler,
                    batch_size=batch_size)
                if not validation_dataset.n_images:
                    raise ValueError(
                            "validate is True, but validation_dataset " + 
                            "contains no images.")
                # this is ideally always the same sequence
                x_val_batch, y_val_batch = validation_dataset.next_batch()
                print("validating with validation set (", validation_dataset.n_images, 
                      "images sequentially) during every summary.")
                
            with tf.name_scope("validation"):
                with tf.variable_scope("model", reuse=True):
                    y_predicted_val_batch = self.model.inference(
                            x_val_batch, is_training=False, print_info=False)
                y_predicted_val_batch = self.postprocess(
                        y_predicted_val_batch, input_shape)
                
                # currently only validates data_loss for performance reasons
                val_loss_single = self.data_loss(y_val_batch, y_predicted_val_batch,
                    loss_fn=loss_fn, cut_loss_to_valid=cut_loss_to_valid)
                # trick with placeholder to be able to calculate val_losses 
                # over entire validation set, before taking the mean
                val_loss_ph = tf.placeholder(tf.float32, name="val_loss")
        else:
            val_loss_single = None
            validation_dataset = None
            val_loss_ph = None

        # get training op
        global_step = tf.train.create_global_step()
        tf.assign(global_step, load_step)
        optimizer = self._define_optimizer(global_step, 
                optimizer_fn=optimizer_fn, learning_rate_fn=learning_rate_fn)
        # extra_update_ops part is needed when using batch_norm (see tf-docs)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            training = optimizer.minimize(total_loss, global_step=global_step)
        
        # define id of new model
        dataset_id = dataset.dataset_id
        run_id = _run_id(ckpt=ckpt,
             load_step=load_step,
             random_seed=random_seed,
             batch_size=batch_size,
             dropout_rate=dropout_rate,
             optimizer_fn=optimizer_fn,
             learning_rate_fn=learning_rate_fn,
             loss_fn=loss_fn,
             cut_loss_to_valid=cut_loss_to_valid,
             weight_reg_str=weight_reg_str,
             weight_reg_fn=weight_reg_fn,
             data_reg_str=data_reg_str,
             data_reg_fn=data_reg_fn,
             comment=comment)
        # run_name is determined depending on existing runs
        # runs are numbered from 0 upwards.

        # get ids of ckpt
        (ckpt_model_id, 
         ckpt_dataset_id, 
         ckpt_run_id, 
         ckpt_run_name) = _parse_ckpt_info(ckpt)
        
        # create folders to store ckpts and logs
        # TODO: The way it is done now can lead to a very deep folder structure.
        # -> Omit saving in subfolders in favor of putting
        #    model_id, run_id etc in a config file
        # TODO: windows paths are restricted to 256 characters!
        run_dir = os.path.join(self.modeldir, self.model_id, ckpt_dataset_id, 
                               ckpt_run_id, dataset_id, run_id)
        #if os.name == 'nt':  # windows
        if len(run_dir) > 200:
            # I think windows restriction is 256 chars
            warn("Path to run_dir is very long. " +
                 "Consider also that the length of windows paths is " +
                 "restricted and that additional chars are added " +
                 "during ckpt generation.")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        run_name = _run_name(run_dir)
        if _exists_in_rundir(run_name, run_dir):  # I think this is redundant.
            raise RuntimeError(
                    "run_name already exists in run_dir. This is a bug. " +
                    "Please fix.  Old ckpt was not overridden.")
        run_dir = os.path.join(run_dir, run_name)
        print("saving new ckpt and logs in", run_dir)
        logdir = os.path.join(run_dir, "logs")
        ckptdir = os.path.join(run_dir, "ckpts")
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if not os.path.exists(ckptdir):
            os.makedirs(ckptdir)
        new_ckpt = os.path.join(ckptdir, run_name)
        
        # op for loading and saving ckpts
        saver = tf.train.Saver(
                max_to_keep=5, keep_checkpoint_every_n_hours=2)

        # op for logs
        loss_summaries = self._loss_summaries(
                total_loss, data_loss, weight_reg_loss, data_reg_loss)
        if validate:
            loss_summaries.append(tf.summary.scalar("val_loss", val_loss_ph))
        image_summaries = self._image_summaries(
                x_batch, y_batch, y_predicted_batch, tf.nn.softmax)
        prediction_summaries = self._prediction_summaries(y_predicted_batch)
        summary = tf.summary.merge(
                [loss_summaries, image_summaries, prediction_summaries])
        writer = tf.summary.FileWriter(logdir, self.sess.graph)
        writer.add_graph(tf.get_default_graph())  # explicitly adding graph

        # load vars from ckpt into model or initialize new model
        if ckpt:
            if self._ckpt_compatible(ckpt_model_id):
                _check_ids(dataset_id, run_id, ckpt_dataset_id, ckpt_run_id)
                self._load_ckpt(saver, ckpt, load_step)  # might throw error here
            else:
                raise ValueError("ckpt is not compatible with model.")
        else:
            self.sess.run(tf.global_variables_initializer())
            
        return (dataset, training, total_loss, saver, new_ckpt, writer, summary, 
                validation_dataset, val_loss_single, val_loss_ph)

    # TODO: make trainer object
    def _train_loop(
            self, n_epochs, batch_size, dataset, total_loss, training, saver, 
            new_ckpt, writer, summary, 
            validation_dataset, val_loss_single, val_loss_ph,
            summary_interval=None, save_interval=None):
        
        validate = validation_dataset is not None

        # tf.data.Dataset makes last batch smaller, if needed.        
        n_batches_per_epoch = ceil(dataset.n_images / batch_size)
        if validate:
            n_val_batches = ceil(validation_dataset.n_images / batch_size)
        n_iterations_per_epoch = n_batches_per_epoch
        if summary_interval is None:
            print("Saving 2 logs per epoch by default.")
            summary_interval = int(0.5*n_iterations_per_epoch) or 1 # iterations
        if save_interval is None:
            print("Saving checkpoint every 2 epochs by default.")
            save_interval = 2*n_iterations_per_epoch  # iterations
        # validation_interval = 2*summary_interval if validate else None

        # training loop        
        # run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True) #DBG
        # self.sess.run(..., options=run_options)  # DBG
        global_step = tf.train.get_global_step()
        step = tf.train.global_step(self.sess, global_step)
        print("starting training with start_step", step)
        for epoch in range(n_epochs):
            print("epoch", epoch+1, "/", n_epochs)
            for iteration in range(n_iterations_per_epoch):
                if step % save_interval == 0:
                    # saves the state before the iteration!
                    print('---->saving', step)
                    saver.save(self.sess, new_ckpt, global_step=step)  # test
                if step % summary_interval == 0: 
                    # summarizes state before iteration!
                    print('---->summarizing', step)
                    if validate:
                        # if step % validation_interval == 0:
                        val_losses = list()
                        print("----> calculating losses on validation set")
                        for _ in range(n_val_batches):
                            val_losses.append(self.sess.run(val_loss_single))
                        val_lossc = np.mean(val_losses)
                        lossc, summaryc, _ = self.sess.run(
                                [total_loss, summary, training], 
                                feed_dict={val_loss_ph:val_lossc})
                        print("---->done", val_losses)
                        print('---->validation loss', val_lossc)
                        # else: 
                        #     lossc, summaryc, _ = self.sess.run(
                        #             [totalloss, summary, training], 
                        #             feed_dict={val_loss_ph:None})
                    else:
                        lossc, summaryc, _ = self.sess.run(
                                [total_loss, summary, training])    
                    writer.add_summary(summaryc, step)
                    print('---->loss', lossc)
                else: # do the same, but without summary
                    lossc, _ = self.sess.run([total_loss, training])
                
                print("iteration", iteration+1, "/", n_iterations_per_epoch)
                step = tf.train.global_step(self.sess, global_step)

        try:
            # cannot ensure that final summary is run because of iterator
            if validate:
                val_losses = list()
                print("---->calculating losses on validation set")
                for _ in range(n_val_batches):
                    val_losses.append(self.sess.run(val_loss_single))
                val_lossc = np.mean(val_losses)
                lossc, summaryc = self.sess.run(
                        [total_loss, summary], feed_dict={val_loss_ph:val_lossc})
                print("---->done", val_losses)
                print('---->validation loss', val_lossc)
            lossc, summaryc = self.sess.run([total_loss, summary])
            writer.add_summary(summaryc, step)                                           
            print('---->summarizing', step, '(final state)')
            print('---->loss', lossc)
            warn("I was expecting to be at the end of sequence here.")
        except tf.errors.OutOfRangeError:
            print("End of sequence")
            print("It is not possible to run the final summary. " +
                  "\nThis behaviour is expected.")
        print('---->saving', step, '(final state)')
        writer.flush()
        saver.save(self.sess, new_ckpt, global_step=step)


    # TODO: pass x_format as format string (?)
    def run_on_image(self, np_x, ckpt, load_step="previous", data_format=None):
        """
        Load model from ckpt and use on an input image np_x.
        
        Args:
            np_x (np-array) : image as np-array.  
                Must be 2D+channel in the form "height-width-channel" or 
                3D+channel in the form "depth-height-width-channel."
            ckpt (str) : path to checkpoint that will be loaded.  
                load_step should not be included in ckpt, but should be provided 
                separately.
            load_step ("previous" or int) : step that will be loaded from ckpt.
                "previous" is converted to the last ckpt that was written
                to the folder containing the ckpt.

        Returns:
            np_y_pred : np-array of the same number of dimensions as np_x.  
                Output size depends on convolution mode of net (same or valid).
        """
        is_training = False
        self._check_data_format(data_format)

        # setup inference
        with tf.name_scope("input_and_preprocessing"):
            x = tf.constant(np_x, dtype=tf.float32, shape=np_x.shape)
            print("input shape:", x.shape)
            input_shape = tf.expand_dims(x, self.model.im_axis).shape
            x_batch = tf.expand_dims(self.preprocess(x), self.model.im_axis)
            
        with tf.variable_scope("model"):
            self.model.set_ready()  # going to load from ckpt
            y_pred = self.model.inference(x_batch, is_training)
        #print("net output_shape:", y_pred.shape)
        y_pred = self.postprocess(y_pred, input_shape)
        y_pred = tf.squeeze(y_pred, axis=self.model.im_axis)
        print("output_shape:", y_pred.shape)
        
        # Load model from ckpt
        saver = tf.train.Saver(max_to_keep=None)
        load_step = _parse_load_step(ckpt, load_step)
        ckpt_model_id, ckpt_dataset_id, ckpt_run_id, ckpt_run_name = \
                _parse_ckpt_info(ckpt)
        if self._ckpt_compatible(ckpt_model_id):        
            self._load_ckpt(saver, ckpt, load_step, False)
        else:
            raise ValueError("ckpt is not compatible with model.")
        
        np_y_pred = self.sess.run(y_pred)
            
        # write graph to tensorboard.  Will show up as . in tensorboard
        run_dir = os.path.join(
                self.modeldir, self.model_id, ckpt_dataset_id, ckpt_run_id,
                ckpt_run_name)
        logdir = os.path.join(run_dir, "logs")
        writer = tf.summary.FileWriter(logdir=logdir, graph=self.sess.graph)
        writer.flush()

        return np_y_pred


    # make loss_fn required arg
    # add weight_reg?
    def test(self, testing_dataset_handler, ckpt, load_step="previous", 
             loss_fn=tf.losses.mean_squared_error, cut_loss_to_valid=False, 
             data_reg_str=None, data_reg_fn=None, batch_size=1):
        """
        Configure network for testing,
        Load model from ckpt
        load testdata
        calculate losses on test set
        
        Args:
            testing_dataset_handler (dataset_handler) : dataset_handler from 
                dataset_handlers.tf_data_dataset_handlers that provides method 
                to get x and y from test set.  DatasetHandlers provide a thin
                layer around tf.data datasets.  If you have an existing tf.data
                dataset, using BaseDatasetHandler is sufficient.  The module also
                provides Handlers that can be initialized from numpy-arrays
                or lists of files. 
            ckpt (str) : path to checkpoint that will be loaded.  
                load_step should not be included in ckpt, but should be provided 
                separately.
            load_step ("previous" or int) : step that will be loaded from ckpt.
                "previous" is converted to the last ckpt that was written
                to the folder containing the ckpt.

            loss_fn (function) : Loss_fn must be a function taking two parameters
                (labels, predictions) and return a loss-tensor.  Common losses can
                be found in tf.losses and in this repo's loss_functions (lf)
                Default: tf.losses.mean_squared_error
            cut_loss_to_valid (bool) :  This can be used for 'same' padding to
                calculate the loss only on that part of the image that is not
                impacted by padding.  This has no impact, if padding='valid'
                Default: False
            data_reg_str (float) : float to scale the strength of data 
                regularization.
            data_reg_fn (function) : a function of a single image returning a loss
                tensor.  Its value is multiplied by data_reg_str and then added
                to the loss.  NOTE that this currently only works correctly for 
                a batch size of 1.
            batch_size (int) : choose how many losses are calculated 
                simultaneously. Use a larger batch size to speed up testing.

        Returns:
            total_loss : (= data_loss + data_reg_loss)
                .
        """       
        
        is_training = False
        
        # load and preprocess dataset        
        with tf.name_scope("training_input_and_preprocessing"):
            dataset = self.TestDatasetAndPreprocess(
                    testing_dataset_handler=testing_dataset_handler, 
                    batch_size=batch_size)
        input_shape = dataset.x_shape
        print("input_shape:", input_shape)
        x_batch, y_batch = dataset.next_batch()
        
        with tf.variable_scope("model"):
            # x_batch is already preprocessed by dataset_handler
            self.model.set_ready()
            y_predicted_batch = self.model.inference(x_batch, is_training)
        y_predicted_batch = self.postprocess(y_predicted_batch, input_shape)
        print("output_shape:", y_predicted_batch.shape)
        
        with tf.name_scope("losses"):
            data_loss_single, _, data_reg_loss_single = self.calculate_losses(
                    y_batch, y_predicted_batch, loss_fn=loss_fn, 
                    cut_loss_to_valid=cut_loss_to_valid,
                    data_reg_str=data_reg_str, data_reg_fn=data_reg_fn)
            total_loss_single = data_loss_single + data_reg_loss_single

        # Load model from ckpt
        saver = tf.train.Saver(max_to_keep=None)
        load_step = _parse_load_step(ckpt, load_step)
        dataset_id = dataset.dataset_id
        ckpt_model_id, ckpt_dataset_id, ckpt_run_id, ckpt_run_name = \
                _parse_ckpt_info(ckpt)
        if self._ckpt_compatible(ckpt_model_id):
            _check_dataset_ids(dataset_id, ckpt_dataset_id)
            self._load_ckpt(saver, ckpt, load_step, False)
        else:
            raise ValueError("ckpt is not compatible with model.")

        print("---->calculating losses on test set")        
        n_batches = ceil(dataset.n_images / batch_size)
        data_losses = list()
        data_reg_losses = list()
        total_losses = list()
        for _ in range(n_batches):
            (data_loss_singlec, 
             data_reg_loss_singlec, 
             total_loss_singlec) = self.sess.run(
                     [data_loss_single,
                      data_reg_loss_single, 
                      total_loss_single])
            data_losses.append(data_loss_singlec)
            data_reg_losses.append(data_reg_loss_singlec)
            total_losses.append(total_loss_singlec)
        # data_lossc = np.mean(data_losses)
        # data_reg_lossc = np.mean(data_reg_losses)
        lossc = np.mean(total_losses)
        print("---->data losses:    ", data_losses)
        print("---->data reg losses:", data_reg_losses)
        print("---->total losses:   ", total_losses)
        print('---->mean total loss:', lossc)
        return lossc


    ## lower level functions
    
    # TODO: this differs from dataset_handler._set_data_format
    def _check_data_format(self, data_format):
        # TODO: change default to the format of model?
        if data_format is None:
            warn("data_format is None.  I am unable to check, if input " +
                 "data_format matches model.data_format.  Set input " +
                 "data_format to avoid this warning.")
        elif data_format in ["channels_last", "channels_first"]:
            if data_format != self.model.data_format:
                raise RuntimeError(
                    "data_format " + data_format + " does not match " +
                    "model.data_format " + self.model.data_format + ".")
        else:
            raise ValueError("Unknown data_format: " + data_format)
    
    # summaries
    def _loss_summaries(self, total_loss, data_loss, weight_reg_loss, 
                        data_reg_loss): 
        summaries = []
        summaries.append(tf.summary.scalar("total_loss", total_loss))
        summaries.append(tf.summary.scalar("data_loss", data_loss))
        summaries.append(tf.summary.scalar("weight_reg_loss", weight_reg_loss))
        summaries.append(tf.summary.scalar("data_reg_loss", data_reg_loss))
        return summaries
    
    # TODO: need to update this to allow different channels and 2d
    # TODO: this is quite data dependent.
    def _image_summaries(self, x_batch, y_batch, y_predicted_batch, activation=None):
        summaries = []
        if len(x_batch.shape) != len(y_batch.shape):
            raise ValueError(
                    "not implemented for the case, where x and y have a " +
                    "different number of dimensions. Detected x_batch.shape: " + 
                    str(x_batch.shape) + " and y_batch.shape: " + 
                    str(y_batch.shape) + ".")
        
        ### TODO: _get_projection_fn(batch) ###
        # need to define a mapping depending on the data domain
        # TODO use _get_channel_axis(...)[0] instead
        if self.model.data_format == "channels_last":
            proj_axis = 1  # z_axis
        elif self.model.data_format == "channels_first":
            proj_axis = 2  # z_axis
        else:
            warn("Will not generate image_summaries, because it is unknown " +
                 "how to handle data_format " + self.model.data_format + ".")
        
        if len(x_batch.shape) == 5:  # 3d input
            x_projection_fn = lambda batch: tf.reduce_max(batch, proj_axis)
        elif len(x_batch.shape) == 4:
            x_projection_fn = lambda batch: batch  # identity
        else:
            raise RuntimeError("x_batch is not 4d and not 5d.")  
              
        if len(y_batch.shape) == 5:  # 3d input
            y_projection_fn = lambda batch: tf.reduce_max(batch, proj_axis)
        elif len(x_batch.shape) == 4:
            y_projection_fn = lambda batch: batch  # identity
        else:
            raise RuntimeError("y_batch is not 4d and not 5d.")       
        
        def pad_color_channel(batch):
            # add empty 3rd color channel
            paddings = [[0,0]] * len(batch.shape)
            paddings[self.model.channel_axis] = [0,1]
            paddings = tf.constant(paddings)
            return tf.pad(batch, paddings)
        if x_batch.shape[self.model.channel_axis] == 2:
            x_color_fn = pad_color_channel
        elif x_batch.shape[self.model.channel_axis] in [1,3]:
            x_color_fn = lambda batch: batch  # identity
        else:
            warn("Will not generate image_summaries, because it is unknown " +
                 "how to handle " + str(x_batch.shape[self.model.channel_axis]) +
                 " color channels.")
            return summaries
        
        if y_batch.shape[self.model.channel_axis] == 2:
            y_color_fn = pad_color_channel                
        elif y_batch.shape[self.model.channel_axis] in [1,3]:
            y_color_fn = lambda batch: batch  # identity
        else:
            warn("Will not generate image_summaries, because it is unknown " +
                 "how to handle " + str(y_batch.shape[self.model.channel_axis]) +
                 " color channels.")
            return summaries
        
        if activation is None:
            activation = lambda batch: batch  # identity
        ### ###

        # TODO: handle the case where output can be negative, e.g. with hinge_loss
        with tf.name_scope("image_summaries"):
#            print(x_batch.shape)
#            print(y_batch.shape)
#            print(y_predicted_batch.shape)
            summaries.append(tf.summary.image(
                    "x", x_projection_fn(x_color_fn(x_batch))))
            summaries.append(tf.summary.image(
                    "y", y_projection_fn(y_color_fn(y_batch))))
            summaries.append(tf.summary.image(
                    "yp", y_projection_fn(y_color_fn(activation(y_predicted_batch)))))
        return summaries

    def _prediction_summaries(self, y_predicted_batch):
        summaries = []
        with tf.name_scope("prediction_summaries"):
            summaries.append(tf.summary.scalar("yp_min", tf.reduce_min(y_predicted_batch)))
            summaries.append(tf.summary.scalar("yp_mean", tf.reduce_mean(y_predicted_batch)))
            summaries.append(tf.summary.scalar("yp_max", tf.reduce_max(y_predicted_batch)))
        return summaries

# need extra summaries for semantic seg -> eg. predictions
#    def _prediction_summaries(self, y_predicted_batch):
#        summaries = []
#        with tf.name_scope("prediction_summaries"):
#            summaries.append(tf.summary.scalar("yp_argmax", tf.argmax(y_predicted_batch)))   
#            summaries.append(tf.summary.scalar("yp_argmax", tf.argmax(y_predicted_batch)))            

    # losses

    def caluclate_total_loss(
            self, y_batch, y_predicted_batch, loss_fn,
            cut_loss_to_valid=False, 
            weight_reg_str=0.0, weight_reg_fn=None, 
            data_reg_str=0.0, data_reg_fn=None):
        """convenience function that calculates losses and adds them up"""
        data_loss, weight_reg_loss, data_reg_loss = self.calculate_losses(
                y_batch=y_batch, y_predicted_batch=y_predicted_batch, 
                loss_fn=loss_fn, cut_loss_to_valid=cut_loss_to_valid, 
                weight_reg_str=weight_reg_str, weight_reg_fn=weight_reg_fn, 
                data_reg_str=data_reg_str, data_reg_fn=data_reg_fn)
        return data_loss + weight_reg_loss + data_reg_loss
    
    def calculate_losses(
            self, y_batch, y_predicted_batch, loss_fn,
            cut_loss_to_valid=False, 
            weight_reg_str=0.0, weight_reg_fn=None, 
            data_reg_str=0.0, data_reg_fn=None):
        """
        returns data_loss, weight_reg_loss, data_reg_loss
        """
        data_loss = self.data_loss(y_batch, y_predicted_batch,
                    loss_fn=loss_fn, cut_loss_to_valid=cut_loss_to_valid)
        if weight_reg_str and weight_reg_fn:
            weight_reg_loss = weight_reg_str*self.regularize_weights(
                    reg_fn=weight_reg_fn, use_mean=True)
        else:
            weight_reg_loss = tf.constant(0.0)
        if data_reg_str and data_reg_fn:
            data_reg_loss = data_reg_str*self.regularize_data(
                    y_predicted_batch, reg_fn=data_reg_fn)
        else:
            data_reg_loss = tf.constant(0.0)
        
        tf.losses.add_loss(data_loss)  # not sure this is right and needed
        tf.losses.add_loss(weight_reg_loss)  # not sure this is right and needed
        tf.losses.add_loss(data_reg_loss)  # not sure this is right and needed
        return data_loss, weight_reg_loss, data_reg_loss

    def data_loss(self, y_batch, y_predicted_batch, loss_fn, 
                  cut_loss_to_valid=False):
        """calculate loss by comparing predicted y with ground truth y"""
        with tf.name_scope("data_loss"):
            if self.model.padding == "same" and cut_loss_to_valid:
                # crop valid part 
                y_batch = self.pretend_network_pass(
                        y_batch, padding="valid", mode="batch")
                y_predicted_batch = self.pretend_network_pass(
                        y_predicted_batch, padding="valid", mode="batch")
            
            print("loss is ", loss_fn.__name__)

            return loss_fn(y_batch, y_predicted_batch)
    
    def regularize_weights(self, reg_fn, use_mean=True):
        """
        reg_fn is a function of a single weight.  All values of 
        reg_fn(weight) are summed.
        
        if use_mean: divide reg_loss by the number of variables in the net.
        This makes reg_str independent of net size. 
        Disable if this takes too long.
        """
        # TODO: there is a bug with use_mean.  Different size models will
        # lead to very different weight losses
        
        # from tf-models-resnet-run_loop
        # batch_normalization variables are excluded from loss.
        def exclude_batch_norm(name):
            return 'batch_norm' not in name
        loss_filter_fn = exclude_batch_norm
        
        with tf.name_scope("weight_regularization"):
            # add weight regularization to the loss
            reg_loss = tf.add_n(
                [reg_fn(v) 
                 for v in tf.trainable_variables() 
                 if loss_filter_fn(v.name)])
            if use_mean:
                print("determining number of trainable vars (except batch_norm) " +
                      "for regularization...")
                n = sum(np.prod(np.array(v.shape.as_list())) 
                        for v in tf.trainable_variables() if loss_filter_fn(v.name))
                print("done: ", n)
                # 1e6 is arbitrary to avoid having to choose very large reg_str
                reg_loss *= (1e6 / n)
            return reg_loss
    
    def regularize_data(self, y_pred_batch, reg_fn):
        """
        reg_fn is a function of a single 3d image
        """
        # TODO: allow 2D
        # TODO: passing fn now.  That fn should handle dimension errors
        # TODO: make reg fn operate on batch
        with tf.name_scope("data_regularization"):
            ndims = len(y_pred_batch.shape)-2
            if ndims != 3 or y_pred_batch.shape[self.model.channel_axis].value > 1:
                raise RuntimeError("tv regularization is only implemented " +
                                   "for single-channel 3D-images.")
            # TODO: channels can be allowed in the same way as batch size
            warn("data regularization is only implemented for batch_size 1. "  +
                 "It will always just use the first image in batch.")
            if self.model.data_format == "channels_last":
                reg_loss = reg_fn(y_pred_batch[0,:,:,:,0])
            elif self.model.data_format == "channels_first":
                reg_loss = reg_fn(y_pred_batch[0,0,:,:,:])
            else:
                raise ValueError(
                        "unexpected data_format", self.model.data_format)
        return reg_loss
        

    def _define_optimizer(self, global_step, learning_rate_fn, optimizer_fn):
        """
        global_step is scalar Variable
        optimizer_fn is a function taking learning rate as parameter, 
            for example: opt_fn = lambda lr: tf.train.AdamOptimizer(lr)
        learning_rate_fn is a function taking global_step as parameter
            for example: lr_fn = lambda global_step: 1e-3
        """
        learning_rate = learning_rate_fn(global_step)
        return optimizer_fn(learning_rate)


    def TrainingDatasetAndPreprocess(
            self, training_dataset_handler, batch_size, n_epochs, 
            random_seed=None):
        return self._DatasetAndPreprocess(
                training_dataset_handler, 
                batch_size, n_epochs, mode="training",
                random_seed=random_seed)

    def ValidationDatasetAndPreprocess(
            self, validation_dataset_handler, batch_size):
        return self._DatasetAndPreprocess(
                dataset_handler=validation_dataset_handler, 
                batch_size=batch_size, n_epochs=None, mode="validation",
                shuffle=False)

    def TestDatasetAndPreprocess(
            self, testing_dataset_handler, batch_size):
        return self._DatasetAndPreprocess(
                dataset_handler=testing_dataset_handler, 
                batch_size=batch_size, n_epochs=None, mode="testing",
                shuffle=False)

    def _DatasetAndPreprocess(
            self, dataset_handler, batch_size, n_epochs, mode=None, 
            random_seed=None, shuffle=True):
        scope = "input_and_preprocessing"
        if mode:
            scope = mode + "_" + scope
        with tf.name_scope(scope):
            # TODO: better: map -> cache -> shuffle_and_repeat -> batch
            dataset = dataset_handler
            if not dataset.n_images:
                warn("dataset contains no images.  I will not preprocess it.")
                return dataset
            if dataset.data_format != self.model.data_format:
                raise RuntimeError(
                        "dataset.data_format " + dataset.data_format + 
                        " does not match model.data_format " +
                        self.model.data_format + ".")
            
            # from tensorflow/models/official/resnet
            # prefetch a batch at a time, This can help smooth out the time taken to
            # load input files as we go through shuffling and processing.
            dataset.prefetch(batch_size)
            # shuffle_buffer: use n_images if possible
            # otherwise 10000 seems to be standard in all examples
            # from tensorflow/models/official/resnet 
            # The buffer size to use when shuffling records. A larger
            # value results in better randomness, but smaller values reduce startup
            # time and use less memory.
            if shuffle:
                dataset.shuffle_and_repeat(buffer_size=dataset.n_images, 
                        n_epochs=n_epochs, random_seed=random_seed, fused=False)
            else:
                dataset._repeat(n_epochs)
            # use num_parallel_calls only in conjunction with batch and map 
            # separately (is also implemented this way)
#            def _preprocess(x,y):
#                """
#                temporarily adding batch dimension for pass to preprocess
#                """
#                x = tf.expand_dims(x, self.model.im_axis)
#                y = tf.expand_dims(y, self.model.im_axis)
#                xpp, ypp = self.preprocess(x, y)
#                xpp = tf.squeeze(x, self.model.im_axis)
#                ypp = tf.squeeze(y, self.model.im_axis)
#                return xpp, ypp
            dataset.map_and_batch(self.preprocess, batch_size, 
                                  num_parallel_calls=batch_size, fused=False)
            if dataset.handler_type == "np":
                print("caching dataset")
                dataset.cache()
            # Operations between the final prefetch and the get_next call to the iterator
            # will happen synchronously during run time. We prefetch here again to
            # background all of the above processing work and keep it out of the
            # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
            # allows DistributionStrategies to adjust how many batches to fetch based
            # on how many devices are present.
            
            # buffer_size=tf.contrib.data.AUTOTUNE --> does not work
            dataset.prefetch(2) # should be ok to just put 1 though
            if dataset.handler_type == "np":
                dataset.make_iterator_and_initialize(self.sess)
            else:
                dataset.make_iterator()
            return dataset

    def preprocess(self, x, y=None):
        with tf.name_scope("preprocessing"):
            if self.normalize_input or self.force_pos:
                # works for both single image and with image batch
                x = self.preprocess_values(x)
            return self.preprocess_shapes(x, y=y, mode="single")

    def postprocess(self, y_predicted_batch, input_shape):
        with tf.name_scope("postprocessing"):
            y_predicted_batch = self.postprocess_shape(y_predicted_batch, input_shape)
            if self.normalize_input or self.force_pos:
                y_predicted_batch = self.postprocess_values(y_predicted_batch)
            return y_predicted_batch

    # mode independent from x
    def preprocess_values(self, x):
        """
        Performs the following preprocessing on x in this order:
        
        - if self.force_pos : Take the square root of x
        - if self.custom_preprocessing_fn : perform custom preprocessing
        - if self.normalize_input : 
          Subtract dataset_means from each pixel and divide by dataset_stds.
          dataset_means and dataset_stds can be scalar or pixel-wise.  If it 
          is pixel-wise, it must have the same shape as a single image in the
          dataset including channels::
              (x - dataset_means) / dataset_stds 
        """
        if not self.normalize_input and not self.force_pos:
            return x
        
        if self.force_pos:
            # take sqrt.  Will square during post-processing
            # TODO: try other forms of force_pos, eg. sigmoid
            x = tf.sqrt(x)
        
        if self.custom_preprocessing_fn is not None:
            x = self.custom_preprocessing_fn(x)
        
        if self.normalize_input:
            if self.force_pos:
                warn("Be aware that you should provide mean and std of " +
                     "the square root of the pixels in the training dataset "
                     "in case of preprocessing.")   
            
            dataset_means = self.dataset_means
            dataset_stds = self.dataset_stds
            
            # TODO: test this            
            if np.ndim(dataset_means) > 0:  # not a scalar
                # dimensions must match except along batch dimension
                if np.any(np.not_equal(
                        dataset_means.shape, x.shape.as_list()[1:])):
                    raise ValueError(
                            "dataset_means is not a scalar and also does not " +
                            "have the same shape as a single input image x." +
                            "dataset_means.shape is " + 
                            str(dataset_means.shape) + " and x.shape is " +
                            str(x.shape) + ".")
            if np.ndim(dataset_stds) > 0:  # not a scalar
                # dimensions must match except along batch dimension
                if np.any(np.not_equal(
                        dataset_stds.shape, x.shape.as_list()[1:])):
                    raise ValueError(
                            "dataset_stds is not a scalar and also does not " +
                            "have the same shape as a single input image x." +
                            "dataset_stds.shape is " + 
                            str(dataset_stds.shape) + " and x.shape is " +
                            str(x.shape) + ".")
            
            x -= dataset_means
            x /= dataset_stds
        return x
    
    
    # move parts of the body to models
    def postprocess_values(self, y_predicted_batch):
        """
        Postprocess y_predicted_batch in this order:
            
        - if self.force_pos : Undo preprocessing: Square y_predicted_batch
        - if self.custom_postprocessing_fn : perform custom postprocessing
        - if self.normalize_input : Undo preprocessing::
            
            y_predicted_batch*dataset_stds + dataset_means
              
        Note:
            This undoes force_pos and normalize_input.
            Custom preprocessing, however, must be explicitly undone by 
            custom_postprocesing.
        """
        if not self.normalize_input and not self.force_pos:
            return y_predicted_batch
        
        if self.custom_postprocessing_fn:
            # TODO: possibly need shape corrections?
            y_predicted_batch = self.custom_postprocessing_fn(y_predicted_batch)
        
        if self.normalize_input:
            if self.force_pos:
                warn("Be aware that you should provide mean and std of " +
                     "the square root of the pixels in the training dataset "
                     "in case of preprocessing.")  
            
            dataset_means = self.dataset_means
            dataset_stds = self.dataset_stds
            
            # TODO: test this            
            if np.ndim(dataset_means) > 0:  # not a scalar
                # dimensions must match except along batch dimension
                if np.any(np.not_equal(
                        dataset_means.shape, 
                        y_predicted_batch.shape.as_list()[1:])):
                    raise ValueError(
                            "dataset_means is not a scalar and also does not " +
                            "have the same shape as a single outpput image " +
                            "y_predicted_batch. dataset_means.shape is " + 
                            str(dataset_means.shape) + 
                            " and y_predicted_batch.shape is " +
                            str(y_predicted_batch.shape) + ".")
                input_shape=dataset_means.shape
                dataset_means = self.preprocess_shapes(dataset_means)
                dataset_means = self.pretend_network_pass(dataset_means, mode="single")                
                dataset_means = self.postprocess_shape(dataset_means, input_shape)
                    
            if np.ndim(dataset_stds) > 0:  # not a scalar
                if np.any(np.not_equal(
                        dataset_stds.shape, 
                        y_predicted_batch.shape.as_list()[1:])):
                    raise ValueError(
                            "dataset_stds is not a scalar and also does not " +
                            "have the same shape as a single output image " +
                            "y_predicted_batch. dataset_stds.shape is " + 
                            str(dataset_stds.shape) + 
                            " and y_predicted_batch.shape is " +
                            str(y_predicted_batch.shape) + ".")
                input_shape=dataset_stds.shape
                dataset_stds = self.preprocess_shapes(dataset_stds)
                dataset_stds = self.pretend_network_pass(dataset_stds, mode="single")                
                dataset_stds = self.postprocess_shape(dataset_stds, input_shape)
        
            y_predicted_batch *= dataset_stds
            y_predicted_batch += dataset_means
        
        if self.force_pos:
            y_predicted_batch = y_predicted_batch**2  
            
        return y_predicted_batch

    # possibly infer mode from x
    # TODO: split functions for x and y in model
    # TODO: is it worth to make extra single method to avoid expand and 
    #       squeeze?
    # TODO why y=None case?
    def preprocess_shapes(
            self, x, y=None, mode="single"):
        """
        give x a shape suitable for net input.
        (see model.preprocess_input_shape for details)
        and give y a shape that corresponds to the net_output_shape
        (during training -> see model.pretend_network_pass)
        """        
        ## PREPROCESSING SHAPE
        # input shape to network is subject to certain constraints depending on
        # padding. The input to a layer in the down-path may not have an odd
        # number of pixels in any spatial dimension or the shapes
        # in up- and down-path will not match during concat.
        
        # This is because, in the up-path, the number of elements in x, y and z 
        # will always be doubled. Thus, it will never be odd.
        # The methods below are implemented to fix the network shape to something
        # hopefully sensible.
    
        # same: pad input on the right with zeroes.
        # valid: cut image to nearest allowed size
        
        # notes to same:
        # max_pooling positive values (with 'same' padding) along an odd-shaped
        # dimension is equivalent to padding with one zero on the right in 
        # that dim and then doing regular max-pooling.
        # If the odd shape occurs in the nth layer.
        # padding with 2**(n-1) zeroes becomes necessary.

        x = self._preprocess_x_shape(x)

        if y is None:
            return x
        
        y = self._preprocess_y_shape(y)

        return x, y

    # TODO: remove mode    
    def _preprocess_x_shape(self, x, mode="single"):
        if mode == "single":  # needed for dataset handler, where mapping is before batch
            x = tf.expand_dims(x, self.model.im_axis)
            x = self.model.preprocess_input_shape(x)
            x = tf.squeeze(x, self.model.im_axis)
        elif mode == "batch":
            x = self.model.preprocess_input_shape(x)
        else:
            raise ValueError(
                    "unknown mode " + str(mode) + ". It can be 'single' or " +
                    "'batch'.")
        return x

    # TODO: remove mode
    def _preprocess_y_shape(self, y, mode="single"):
        if mode == "single":  # needed for dataset handler, where mapping is before batch
            y = tf.expand_dims(y, self.model.im_axis)
            y = self.model.pretend_network_pass(y)
            y = tf.squeeze(y, self.model.im_axis)
        elif mode == "batch":
            y = self.pretend_network_pass(y)
        else:
            raise ValueError(
                    "unknown mode " + str(mode) + ". It can be 'single' or " +
                    "'batch'.")
        return y
    
    # move function body to utils
    def postprocess_shape(self, y_predicted_batch, input_shape):
        """if the input for "same" was padded, remove padding again."""
        if self.model.padding == "same":    
            print("cropping net ouput to original shape " + str(input_shape))
            input_shape = na.utils.convert_shape_to_np_array(input_shape)
            with tf.name_scope("fix_output_size"):
                ndims = len(input_shape)-2
                spatial_axes = na.utils.get_spatial_axes(
                        mode="batch", ndims=ndims,
                        im_axis=self.model.im_axis, 
                        channel_axis=self.model.channel_axis)
                spatial_input_shape = input_shape[spatial_axes]
                # TODO: implement this using slice()
                if ndims == 3:
                    if self.model.channel_axis == 1:
                        y_predicted_batch = y_predicted_batch[
                                :,
                                :,
                                :spatial_input_shape[0],
                                :spatial_input_shape[1],
                                :spatial_input_shape[2]]
                    elif self.model.channel_axis == -1:
                        y_predicted_batch = y_predicted_batch[
                                :,
                                :spatial_input_shape[0],
                                :spatial_input_shape[1],
                                :spatial_input_shape[2],
                                :]
                    else:
                        raise NotImplementedError(
                                "postprocess shapes has not been implemented for " +
                                str(self.model.data_format) + ".")
                elif ndims == 2:
                    if self.model.channel_axis == 1:
                        y_predicted_batch = y_predicted_batch[
                                :,
                                :,
                                :spatial_input_shape[0],
                                :spatial_input_shape[1]]
                    elif self.model.channel_axis == -1:
                        y_predicted_batch = y_predicted_batch[
                                :,
                                :spatial_input_shape[0],
                                :spatial_input_shape[1],
                                :]
                    else:
                        raise NotImplementedError(
                                "postprocess shapes has not been implemented for " +
                                str(self.model.data_format) + ".")                

        # do nothing for valid padding
        return y_predicted_batch
    
    ##
    def pretend_network_pass(self, x, override_padding=None, exclude_channel_axis=True):
        return self.model.pretend_network_pass(
                x, 
                override_padding=override_padding,
                exclude_channel_axis=exclude_channel_axis)
    
    def get_net_output_shape(self, input_shape):
        return self.model._get_net_output_shape(
                input_shape, input_shape[self.model.channel_axis])

    def _ckpt_compatible(self, ckpt_model_id):
        if ckpt_model_id != self.model_id:
            raise ValueError("Cannot load from ckpt. model_id does not " +
                             "match ckpt_model_id.")
        return True

    def _load_ckpt(self, saver, ckpt, load_step, need_to_change_seed=True):
        """
        use saver to restore ckpt in self.sess
        
        Args:
            ckpt (str) : name of ckpt (without step number)
            load_step (int) : step number of ckpt.
            need_to_change_seed (bool) : issue a warning, if this is
                not explicitly set to False.
        """
        
        if ckpt and not load_step is None:  # if ckpt != None, 0, ""
            if need_to_change_seed:
                warn("You should change random seed, when loading from ckpt " +
                 "during training.  Set need_to_change_seed in _load_ckpt to " +
                 "False to avoid this warning.")
            ckpt_name = ckpt + "-" + str(load_step)
            # print("loading ckpt:", ckpt_name)  # this info is printed by tf
            saver.restore(self.sess, ckpt_name)
        elif ckpt:
            # TODO: improve error message and error handling
            raise ValueError("ckpt was provided, but no load_step.")
        elif load_step:
            raise ValueError("load_step was provided, but no ckpt.")
        else:
            raise ValueError("Neither ckpt nor load_step were provided in call" + 
                             "of _load_ckpt.")
        return load_step
    
    def _set_model_id(self, comment=None):
        """
        combines all args to NDNet.__init__ to a string
        """
        # arch
        mid = self.arch
        mid += "_" + self.model.padding
        # net features
        mid += "_fp" + str(int(self.force_pos))
        pp = (str(2) if self.custom_preprocessing_fn
                     else str(int(self.normalize_input)))
        mid += "_pp" + pp
        # training features
        bn = (str(2) if self.model.use_batch_renorm 
                     else str(int(self.model.use_batch_norm)))
        mid += "_bn" + bn
        mid += str(int(self.model.last_layer_batch_norm))
        if self.model.data_format == "channels_first":
            mid += "_chfirst"
        elif self.model.data_format == "channels_last":
            mid += "_chlast"
#        mid += "_ca" + str(int(self.model.channel_axis))
        if comment is not None:
            mid += "_" + comment
        return mid

## END CLASS
def _parse_ckpt_info(ckpt):
    if ckpt is None:
        return "", "", "", ""
    ckpt_head, _ = os.path.split(ckpt) # filename is also ckpt_run_name
    ckpt_head, _ = os.path.split(ckpt_head)
    ckpt_head, ckpt_run_name = os.path.split(ckpt_head)
    ckpt_head, ckpt_run_id = os.path.split(ckpt_head)
    ckpt_head, ckpt_dataset_id = os.path.split(ckpt_head)
    ckpt_head, ckpt_model_id = os.path.split(ckpt_head)
    return ckpt_model_id, ckpt_dataset_id, ckpt_run_id, ckpt_run_name

def _check_ids(dataset_id, run_id, ckpt_dataset_id, ckpt_run_id):
    _check_dataset_ids(dataset_id, ckpt_dataset_id)
    _check_run_ids(run_id, ckpt_run_id)

def _check_dataset_ids(dataset_id, ckpt_dataset_id):
    if ckpt_dataset_id == "unknown_dataset":
        warn("ckpt_dataset_id was not set.  It is unknown, which " +
             "dannntaset was used to train the model.")
    elif ckpt_dataset_id != dataset_id:
        print("ckpt was originally trained with ckpt_dataset_id", ckpt_dataset_id, 
              ".\n It is now run with dataset_id", dataset_id)
        
def _check_run_ids(run_id, ckpt_run_id):
    if ckpt_run_id != run_id:
        print("ckpt was originally trained with ckpt_run_id", ckpt_run_id,
              ".\n It is now run with with run_id", run_id)

# there is also saver.last_checkpoints and tf.train.last_checkpoints 
# to deal with "previous" situation
# but wouldn't that require saver to know ckpt path?
def _parse_load_step(ckpt, load_step):
    if ckpt:
        if load_step == "previous":
            ckpt_sub, _ = os.path.split(ckpt)
            with open(os.path.join(ckpt_sub, "checkpoint"), "r") as f:
                ckpt_files = list(csv.reader(f, delimiter=":"))
            old_ckpt_name = ckpt_files[0][1][2:-1]  # name of first in list
            dashpos = old_ckpt_name[::-1].index('-')
            load_step = int(old_ckpt_name[-dashpos:])
        elif load_step is None:
            raise ValueError("load step cannot be None, if ckpt is given.")
    else:
        if load_step:
            warn("load step is ignored as ckpt is None.")
        load_step = 0  # -> remove in favor of global step   
    return load_step

# TODO: let higher level fn deal with this
def _run_id(ckpt, load_step, random_seed, batch_size, dropout_rate,
            optimizer_fn, learning_rate_fn, loss_fn, cut_loss_to_valid, 
            weight_reg_str, weight_reg_fn, data_reg_str, data_reg_fn, 
            comment=None):
    if ckpt == None:
        # rid = "init_" + weight_init.__name__ + "_seed" + str(random_seed) (?)
        rid = "seed" + str(random_seed)
    else:
        rid = "loadstep" + str(load_step) + "_newseed" + str(random_seed)
    rid += "_bs" + str(batch_size)
    rid += "_do" + str(dropout_rate) # TODO: specify scientific format
    #rid += "_opt" + # how to do this intelligently?  Maybe just let user do own comment
    #rid += "_lr" + # how to do this intelligently?  Maybe log lr in summaries
    rid += "_loss=" + loss_fn.__name__ + str(int(cut_loss_to_valid))
    wreg_name = weight_reg_fn.__name__ if weight_reg_fn is not None else "None"
    dreg_name = data_reg_fn.__name__ if data_reg_fn is not None else "None"
    rid += "_weightreg=" + str(weight_reg_str) + wreg_name
    rid += "_datareg=" + str(data_reg_str) + dreg_name
    if comment:
        rid += "_" + comment
    return rid

def _run_name(run_dir):
    run_string = "run"
    # if os.path.isdir(run_dir):  # -> No, let the system handle FileNotFoundError
    subfolders = [f.name for f in os.scandir(run_dir) if f.is_dir()]
    # more safe would be to check for contents
    run_folders = [f for f in subfolders if f[:3] == "run"]
    if not run_folders:
        run_number = "0"
    else:
        run_number_exists = [f[3:] for f in run_folders if f[3:].isdigit()]
        run_number_exists.sort(key=int)
        run_number = str(int(run_number_exists[-1]) + 1)
    return run_string + run_number
    

def _exists_in_rundir(run_name, run_dir):
    """check if folder named run_name already exists in run_dir"""
    # if os.path.isdir(run_dir):  # -> No, let the system handle FileNotFoundError
    subfolders = [f.name for f in os.scandir(run_dir) if f.is_dir()]
    return run_name in subfolders


# %% convenience functions
# TODO: remove optimizer_fn, learning_rate_fn, etc.

# e.g. sess_config=tf.ConfigProto(log_device_placement=True)
def _train(
        training_data_path, n_epochs, batch_size, 
        # net features
        arch="unetv3", padding="same", force_pos=False, normalize_input=False,
        use_batch_norm=False,
        # model loading
        ckpt=None, load_step=None,
        # training features
        dropout_rate=0.0, 
        optimizer_fn=lambda lr : tf.train.AdamOptimizer(lr),
        learning_rate_fn=lambda training_step: 1e-3,
        # loss function
        loss_fn=tf.losses.mean_squared_error, cut_loss_to_valid=False, 
        weight_reg_str=None, data_reg_str=None, data_reg_fn=None,
        # details
        random_seed=None, validate=False, run_comment=None, sess_config=None):
    """TODO"""
    
    data_format="channels_last"

    # TODO: generate dataset specs here, if they don't exist
    
    # for IPHT data
    training_dataset_handler = dh.tfdata_dataset_handlers.VascuPairsListDatasetHandler(
            training_data_path, mode="train", data_format=data_format)
    if validate:
        validation_dataset_handler = dh.tfdata_dataset_handlers.VascuPairsListDatasetHandler(
                training_data_path, mode="validation", data_format=data_format)
    else:
        validation_dataset_handler=None
    
    x_ch = training_dataset_handler.get_x_channels()
    y_ch = training_dataset_handler.get_y_channels()
    net_channel_growth = int(round(y_ch/x_ch))
    
    if normalize_input:
        scalar_mean = training_dataset_handler.mean
        scalar_std = training_dataset_handler.std
        if scalar_mean is None or scalar_std is None:
            raise RuntimeError("_train is not functioning correctly.")
    else:  # To avoid warning
        scalar_mean = None
        scalar_std = None

    try:
        with tf.Session(config=sess_config) as sess:
            deconv_net = NDNet(
                    sess=sess, 
                    arch=arch, 
                    padding=padding,
                    force_pos=force_pos,
                    normalize_input=normalize_input,
                    use_batch_renorm=False, # not very useful
                    use_batch_norm=use_batch_norm, 
                    last_layer_batch_norm=None, # True if any is used
                    data_format=data_format, # last
                    dataset_means=scalar_mean,
                    dataset_stds=scalar_std,
                    _net_channel_growth=net_channel_growth,
                    comment=None)
            deconv_net.train(
                    training_dataset_handler=training_dataset_handler, 
                    n_epochs=n_epochs, 
                    batch_size=batch_size, 
                    learning_rate_fn=learning_rate_fn,
                    optimizer_fn=optimizer_fn, 
                    loss_fn=loss_fn, 
                    cut_loss_to_valid=cut_loss_to_valid, 
                    weight_reg_str=weight_reg_str, 
                    weight_reg_fn=tf.nn.l2_loss, # rarely changed
                    data_reg_str=data_reg_str, 
                    data_reg_fn=data_reg_fn,
                    ckpt=ckpt, 
                    load_step=load_step, 
                    random_seed=random_seed, 
                    weight_init=training_utils.he_init, 
                    batch_renorm_fn=None,
                    dropout_rate=dropout_rate,
                    validate=validate,
                    validation_dataset_handler=validation_dataset_handler,
                    summary_interval=None, 
                    save_interval=None,
                    comment=run_comment)
    
        tf.reset_default_graph()
    except Exception as e:
        tf.reset_default_graph()
        raise e
    print("done")
    print(datetime.datetime.now().time())


def _run_on_image(
        np_x, ckpt, load_step="previous", x_format="dhwc",
        # network params
        arch="unetv3", padding="same", force_pos=False, normalize_input=False, 
        use_batch_norm=False, net_channel_growth=1,
        # calculate loss on testimage (except weight_reg_str)
        cal_loss=False, np_y=None, y_format=None,
        loss_fn=lf.regression.l2loss, cut_loss_to_valid=False, 
        data_reg_str=None, data_reg_fn=None,
        # details
        dataset_means=None, dataset_stds=None, sess_config=None):
    """
    runs inference on an image
    
    Args:
    
    np_x (np-array) : input image to the network.  Shape can be 
        3D+channels, 3D, 2D+channels or 2D.  
        NOTE: Use format string to specify the meaning of each dimension.  
        It will be converted automatically to the default input format to 
        the network ("dhwc" or "hwc" resp.).
    
    x_format (str) : format string 
        The format string defines what each axis in np_x stands for.
        Allowed for 3D images is any permutation of:
        - "dhwc" (depth, height, width, channels) 
        - "dhw" (no channel axis, eg. black/white image) 
        For 2D, you should leave out "d"
        Default: "dhwc" 
        
    TODO
    """
    data_format = "channels_last"
    
    # parse format string
    np_x = _parse_format_string(np_x, x_format)  # returns channels_last
    try:
        with tf.Session(config=sess_config) as sess:
            deconv_net = NDNet(
                    sess=sess, 
                    arch=arch, 
                    padding=padding,
                    force_pos=force_pos,
                    normalize_input=normalize_input,
                    use_batch_renorm=False, # not very useful
                    use_batch_norm=use_batch_norm, 
                    last_layer_batch_norm=None, # True if any is used
                    data_format=data_format, # last
                    dataset_means=dataset_means,
                    dataset_stds=dataset_stds,
                    _net_channel_growth=net_channel_growth)

            np_y_pred = deconv_net.run_on_image(np_x, ckpt, load_step=load_step)
            
            if cal_loss:
                print("----> calculating loss on image") 
                if np_y is None:
                    raise ValueError(
                            "You need to provide np_y, if cal_loss is True.")
                
                if y_format is None:
                    y_format = x_format
                np_y = _parse_format_string(np_y, y_format)  # returns channels_last
                if np_y.shape != np_x.shape:
                    raise ValueError(
                            "np_x and np_y must have the same shape. " +
                            "After parsing format string, " + 
                            "np_x has shape " + str(np_x.shape) + " " +
                            "and np_y has shape " + str(np_y.shape) + ".")
                
                y_predicted_batch = tf.constant(
                        np.expand_dims(np_y_pred, deconv_net.model.im_axis))
                y_batch = tf.constant(
                        np.expand_dims(np_y, deconv_net.model.im_axis))
                y_batch = deconv_net.pretend_network_pass(
                        y_batch, exclude_channel_axis=True)
                
                with tf.name_scope("losses"):
                    data_loss, _, data_reg_loss = deconv_net.calculate_losses(
                            y_batch, y_predicted_batch, loss_fn=loss_fn, 
                            cut_loss_to_valid=cut_loss_to_valid,
                            data_reg_str=data_reg_str, data_reg_fn=data_reg_fn)
                    total_loss = data_loss + data_reg_loss
                       
                (data_lossc, data_reg_lossc, total_lossc) = deconv_net.sess.run(
                         [data_loss, data_reg_loss, total_loss])
                print("---->data loss:    ", data_lossc)
                print("---->data reg loss:", data_reg_lossc)
                print("---->total loss:   ", total_lossc)

        tf.reset_default_graph()
    except Exception as e:
        tf.reset_default_graph()
        raise e

    return np_y_pred


# note that _train, _run_on_image and _test always use channels_last
def _parse_format_string(np_x, x_format):
    """flips axes of np_x so that output has format dhwc or hwc"""
    if x_format in ["dhwc", "hwc"]:
        return np_x

    ndim = np_x.ndim
    if ndim not in [2,3,4]:
        raise ValueError(
                "This is only implemented for 2D, 2D+channels, 3D, " +
                "3D+channels.  Allowed np_x.ndim are thus 2,3,4.")
    
    allowed_basic_formats = ["dhwc", "dhw", "hwc", "hw"]
    if (x_format not in [''.join(el) for el in permutations(allowed_basic_formats[0])] and
        x_format not in [''.join(el) for el in permutations(allowed_basic_formats[1])] and
        x_format not in [''.join(el) for el in permutations(allowed_basic_formats[2])] and
        x_format not in [''.join(el) for el in permutations(allowed_basic_formats[3])]):
        raise ValueError(
                "Invalid format string. Allowed are all permutations of " + 
                "entries in " + str(allowed_basic_formats) + ".")

    if len(x_format) != ndim:
        raise ValueError(
                "Format string is valid, but does not match np_x. " +
                "Format string must have length np_x.ndim")
    
    if "d" in x_format:
        d_axis = x_format.index("d")
    
    if "c" in x_format:
        c_axis = x_format.index("c")
    else:
        c_axis = ndim  # channels last
        np.expand_dims(np_x, c_axis)
        ndim += 1
        
    h_axis = x_format.index("h")
    w_axis = x_format.index("w")
    
    if ndim == 4:
        perm = [d_axis, h_axis, w_axis, c_axis]
    elif ndim == 3:
        perm = [h_axis, w_axis, c_axis]
        
    return np_x.transpose(perm)


def _test(
        data_path, ckpt, load_step="previous",
        # network params
        arch="unetv3", padding="same", force_pos=False, normalize_input=False, 
        use_batch_norm=False,
        # loss
        loss_fn=lf.regression.l2loss, cut_loss_to_valid=False, 
        data_reg_str=None, data_reg_fn=None,
        # details
        batch_size=1, dataset_means=None, dataset_stds=None, sess_config=None):
    """TODO"""
    
    data_format="channels_last"
    
    # for IPHT data
    dataset_handler = dh.tfdata_dataset_handlers.VascuPairsListDatasetHandler(
            data_path, mode="test", data_format=data_format)
    
    x_ch = dataset_handler.get_x_channels()
    y_ch = dataset_handler.get_y_channels()
    net_channel_growth = int(round(y_ch/x_ch))
    
    try:
        with tf.Session(config=sess_config) as sess:
            deconv_net = NDNet(
                    sess=sess, 
                    arch=arch, 
                    padding=padding,
                    force_pos=force_pos,
                    normalize_input=normalize_input,
                    use_batch_renorm=False, # not very useful
                    use_batch_norm=use_batch_norm, 
                    last_layer_batch_norm=None, # True if any is used
                    data_format=data_format, # last
                    dataset_means=dataset_means,
                    dataset_stds=dataset_stds,
                    _net_channel_growth=net_channel_growth)
            lossc = deconv_net.test(
                    dataset_handler, ckpt, load_step=load_step, 
                    loss_fn=loss_fn, cut_loss_to_valid=cut_loss_to_valid, 
                    data_reg_str=data_reg_str, data_reg_fn=data_reg_fn, 
                    batch_size=batch_size)
        tf.reset_default_graph()
    except Exception as e:
        tf.reset_default_graph()
        raise e
    print("done")
    print(datetime.datetime.now().time())
    return lossc



# %% main

def test_helpers():
    ##### TEST TRAINING
    # example: train for 1 epoch, then run_on_image
    print("testing training")
    start = datetime.datetime.now()
    print(start)
    _train(training_data_path="testdata", n_epochs=2, batch_size=1, 
           # net features
           arch="unetv3", padding="valid", force_pos=False, 
           normalize_input=False, use_batch_norm=False, 
           # model loading
           ckpt=None, load_step=None, 
           # training features
           dropout_rate=0.0, 
           learning_rate_fn=lambda training_step: 1e-4,
           optimizer_fn=lambda lr : tf.train.AdamOptimizer(lr), 
           # loss function
           loss_fn=lf.regression.l2loss, cut_loss_to_valid=False, 
           weight_reg_str=0.001, data_reg_str=None, data_reg_fn=None,
           # details
           random_seed=1, validate=False, 
           run_comment="test_conenience", sess_config=None)
    end = datetime.datetime.now()
    print("done")
    print(end)
    print("elapsed time: ", end-start)

    ##### TEST _RUN_ON_IMAGE
    # these ids result from the above example call to _train
    # will always load run0 !
    # read the printout from start of training to know it
    print("\ntesting run_on_image")
    model_id = "unetv3_valid_fp0_pp0_bn00_chlast"
    dataset_id = "poisson_n1000_wl520"
    run_id = ("seed1_" +
              "bs1_do0.0_" + 
              "loss=l2loss0_weightreg=0.001l2_loss_datareg=NoneNone_" +
              "test_convenience")
    run_number = "run0"  
    ckpt = os.path.join(
            "models", model_id, dataset_id, run_id, run_number, "ckpts", 
            run_number)
    np_x = dh.dataset_utils.np_load(
            os.path.join("testdata","image0","im.mat"), expand_dims=-1, 
            version='v7')
    np_y = dh.dataset_utils.np_load(
            os.path.join("testdata","image0","obj.mat"), expand_dims=-1, 
            version='v7')
    _run_on_image(
            np_x, ckpt, load_step="previous", x_format="dhwc",
            # net
            arch="unetv3", padding="valid", force_pos=False, 
            normalize_input=False, use_batch_norm=False, net_channel_growth=1,
            # cal_loss
            cal_loss=True, np_y=np_y, y_format="dhwc",
            loss_fn=lf.regression.l2loss, cut_loss_to_valid=False, 
            data_reg_str=None, data_reg_fn=None,
            # details
            dataset_means=None, dataset_stds=None, sess_config=None)
    training_utils.print_array_info(np_y, "result")

    ##### TEST TESTING
    _test(data_path="testdata", ckpt=ckpt, load_step="previous",
          # network params
          arch="unetv3", padding="valid", force_pos=False, 
          normalize_input=False, use_batch_norm=False,
          # loss
          loss_fn=lf.regression.l2loss, cut_loss_to_valid=False, 
          data_reg_str=None, data_reg_fn=None,
          # details
          batch_size=1, dataset_means=None, dataset_stds=None, 
          sess_config=None)
    
    
def main():
    ##### TEST TRAINING
    # example: train for 1 epoch, then run_on_image
    
    # UPDATE: using small model for quick testing on my laptop with cpu only 
    # (memory <8GB, time < 1 min)
    
    # make sure you have enough GPU-memory.  unet_v3 requires about 10 GB for
    # 400x100x100x1 image input
    # running it takes about 10 minutes on my sytem (TitanX Maxwell)
    # will create a folder "models" to store ckpts and logs
    
    # TODO: allow passing constant or optimizer and do this in code
    # TODO: add extra learning_rate_decay/learning_rate_decay_fn argument
    
    print("testing training")
    # data input:
    x_filelist = [os.path.join("testdata", "image0", "im.mat")]
    y_filelist = [os.path.join("testdata", "image0", "obj.mat")]
    def load_data_pair(x_path, y_path):
        """function to load a single image pair from their respective paths."""
        # expand dims is used, because NDNet requires images to have
        # a channel dimension, but testimages do not contain it.
        # expand_dims=-1, because NDNet is used with "channels_last"
        # below.
        np_x = dh.dataset_utils.np_load(x_path, expand_dims=-1, version='v7')
        np_y = dh.dataset_utils.np_load(y_path, expand_dims=-1, version='v7')
        # load_fn must explicitly return float32!
        # TODO: improve error handling in net if this is not the case
        return np_x.astype(np.float32), np_y.astype(np.float32)
    dataset_id = "poisson_n1000_wl520"  # this may be defined to id dataset.
    
    training_dataset_handler = dh.tfdata_dataset_handlers.BaseListDatasetHandler(
            x_filelist,
            y_filelist,
            load_data_pair,
            dataset_id=dataset_id,
            data_format="channels_last")  # channels first is not well tested

    start = datetime.datetime.now()
    print(start)
    # sess_config=tf.ConfigProto(log_device_placement=True) to DBG memory error
    try: 
        with tf.Session() as sess:
            deconv_net = NDNet(
                    sess=sess, 
                    arch="unetv3_small",
                    padding="valid",
                    force_pos=False,
                    normalize_input=False,
                    use_batch_renorm=False,  # I don't recommend using this any more.
                    use_batch_norm=False, 
                    last_layer_batch_norm=None, # True if bn or brn is used
                    data_format="channels_last",  # "channels_first" not well tested
                    dataset_means=None,  # would be needed for normalize_input=True
                    dataset_stds=None)  
            deconv_net.train(
                    training_dataset_handler=training_dataset_handler, 
                    n_epochs=2, 
                    batch_size=1,  # TODO: handle case where batch_size > n_images
                    ckpt=None,  # starts from 0.  Else give path to existing ckpt and ...
                    load_step=None,  # ... provide load_step as well
                    learning_rate_fn=lambda global_step: 1e-4,  # constant
                    optimizer_fn=lambda lr: tf.train.AdamOptimizer(lr), 
                    loss_fn=lf.regression.l2loss, # == tf.losses.mean_squared_error
                    cut_loss_to_valid=False, 
                    weight_reg_str=0.001, 
                    weight_reg_fn=tf.nn.l2_loss,
                    data_reg_str=0.00000001,
                    data_reg_fn=None,
                    random_seed=1, 
                    weight_init=training_utils.he_init,  # == tf.initializers.he_normal
                    batch_renorm_fn=None,
                    dropout_rate=0.0,
                    validate=False,
                    validation_dataset_handler=None,
                    comment="example")
    except Exception as e:
        tf.reset_default_graph()
        raise e
    end = datetime.datetime.now()
    print("done")
    print(end)
    print("elapsed time: ", end-start)
    tf.reset_default_graph()
    # will close session later.  reusing net now.
    

    ##### TEST RUN_ON_IMAGE
    # these ids result from the above example call to train
    # read the printout from start of training to know it
    print("\ntesting run_on_image")
    model_id = "unetv3_small_valid_fp0_pp0_bn00_chlast"
    run_id = ("seed1_" +
              "bs1_do0.0_" + 
              "loss=l2loss0_weightreg=0.001l2_loss_datareg=1e-08None_" +
              "example")
    run_number = "run0"  # will always load first run of example !
    
    # load ckpt from model_id, dataset_id (from above), run_id and run_number
    ckpt = os.path.join(
            "models", model_id, dataset_id, run_id, run_number, "ckpts", 
            run_number)
    np_x = dh.dataset_utils.np_load(x_filelist[0], expand_dims=-1, version='v7')
    try:
        with tf.Session() as sess:
            deconv_net = NDNet(
                    sess=sess, 
                    arch="unetv3_small", 
                    padding="valid",
                    force_pos=False,
                    normalize_input=False,
                    use_batch_renorm=False,  # I don't recommend using this any more.
                    use_batch_norm=False, 
                    last_layer_batch_norm=None, # True if bn or brn is used
                    data_format="channels_last",  # "channels_first" not well tested
                    dataset_means=None,  # would be needed for normalize_input=True
                    dataset_stds=None)
            # TODO: currently it is not possible to simply use NDNet instance
            # from above.
            # This applies to both loading ckpt into this instance and
            # using the model params already existing in model.
            # TODO: change this
            np_y = deconv_net.run_on_image(
                    np_x, ckpt, "previous", data_format="channels_last")
    except Exception as e:
        tf.reset_default_graph()
        raise e
    tf.reset_default_graph()
    # sess.close()
    training_utils.print_array_info(np_y, "result")

    ##### TEST TESTING
    # these ids result from the above example call to train
    # read the printout from start of training to know it
    print("\ntesting 'test'-method.")
    model_id = "unetv3_small_valid_fp0_pp0_bn00_chlast"
    run_id = ("seed1_" +
              "bs1_do0.0_" + 
              "loss=l2loss0_weightreg=0.001l2_loss_datareg=1e-08None_" +
              "example")
    run_number = "run0"  # will always load first run of example !
    
    # load ckpt from model_id, dataset_id (from above), run_id and run_number
    ckpt = os.path.join(
            "models", model_id, dataset_id, run_id, run_number, "ckpts", 
            run_number)
    # of course you would change this to a different filelist in a real setting
    testing_dataset_handler = dh.tfdata_dataset_handlers.BaseListDatasetHandler(
        x_filelist,
        y_filelist,
        load_data_pair,
        dataset_id=dataset_id,
        data_format="channels_last")
    try:
        with tf.Session() as sess:
            deconv_net = NDNet(
                    sess=sess, 
                    arch="unetv3_small", 
                    padding="valid",
                    force_pos=False,
                    normalize_input=False,
                    use_batch_renorm=False,  # I don't recommend using this any more.
                    use_batch_norm=False, 
                    last_layer_batch_norm=None, # True if bn or brn is used
                    data_format="channels_last",  # "channels_first" not well tested
                    dataset_means=None,  # would be needed for normalize_input=True
                    dataset_stds=None)
            # - TODO: currently it is not possible to simply use NDNet instance
            #   from above.
            #   This applies to both loading ckpt into this instance and
            #   using the model params already existing in model.
            # - TODO: change this
            # - test returns mean total loss
            deconv_net.test(
                    testing_dataset_handler,  
                    ckpt,
                    "previous",
                    loss_fn=lf.regression.l2loss, # as above
                    cut_loss_to_valid=False,
                    data_reg_str=0.00000001,
                    data_reg_fn=None,
                    batch_size=1)  # have not tested batch_size>1
    except Exception as e:
        tf.reset_default_graph()
        raise e
    tf.reset_default_graph()
    
if __name__ == '__main__':
    #test_helpers()
    main()
