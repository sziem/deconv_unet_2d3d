#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:14:22 2018

@author: Soenke
"""

# %% conv layers
@deprecated("the recommended function is upsample_NN.")
def upsample_conv_layerv3(inputs, out_num, upsam_size, padding, is_training, scope,
               conv_size, 
               use_batch_renorm=False, use_batch_norm=False, use_dropout=False, 
               activation="default", init="default", batch_renorm_scheme=None,
               data_format='channels_last'
    # conv_size is conv_kernel_size
    # bn->conv->relu
    
    # parsing input
    # TODO: there was a nice function to set this (in the former layer)
    # in the resnet-code
    if data_format == 'channels_last' or 'NDHWC':  # recommended for CPU
        axis = -1 
    elif data_format == 'channels_first' or 'NCDHW':  # recommended for GPU
        axis = 1
        
    if activation == "default":
        activation = tf.nn.relu  # tf.nn.leaky_relu, tf.nn.elu
    if init == "default":
        init = he_init
        
    if not (use_batch_norm or use_batch_renorm):
        use_bias = True
    else:
        use_bias = False

    # dr kwargs: rate
    # bn kwargs: momentum, epsilon
    # brn kwargs: momentum, epsilon, renorm_momentum, renorm_clipping
    
    # TODO: not sure where it is customary to put this.
    if use_dropout:  # dropout + batch_norm = :(
        outs = tf.layers.dropout(outs, 
                                 rate=0.05, 
                                 training=is_training, 
                                 name=scope+'/dropout')

    # I think batch_norm could just be replaced by subtracting population mean
    # or any number suitable to the scale of activations to make input to 
    # conv layer not all positive
    if use_batch_renorm:  
        if use_batch_norm:
            warn(scope + ": Both batch_renorm and batch_norm are True. " +
                 "Will use batch_renorm.")
        if not batch_renorm_scheme:
            dict(rmin=0, rmax=np.inf, dmax=np.inf)  # default, no clipping
        outs = tf.layers.batch_normalization(outs,
                        renorm=True,
                        renorm_clipping=batch_renorm_scheme,
                        renorm_momentum=0.95,  # default: 0.99
                        axis, # related to data_format of conv
                        momentum=0.99,  # default
                        epsilon=1e-3,  # default 1e-3
                        training=is_training,
                        name=scope+'/batch_norm',
                        fused=True)
    elif use_batch_norm:
            #TODO: try regularizing beta and gamma (no?)
            outs = tf.layers.batch_normalization(outs, 
                        axis = -1, # related to data_format of conv
                        momentum=0.99,  # default
                        epsilon=1e-3,  # default
                        training=is_training,
                        name=scope+'/batch_norm',
                        fused=True)   
    
    outs = tf.layers.conv3d_transpose(inputs,
                        out_num,
                        kernel_size,
                        strides=upsam_size,
                        padding=padding,
                        data_format=data_format, # related to axis down
                        activation=activation,
                        use_bias=use_bias,  # not needed whens using batch_norm
                        kernel_initializer=tf.truncated_normal_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name=scope+'/conv')
    return outs

@deprecated("the recommended function is conv_layer")
def conv_layer_original(inputs, out_num, kernel_size, padding, is_training, scope):
    """as in zhenyang wang's implementation."""
    shape = list(kernel_size) + [inputs.shape[-1].value, out_num]  
    # + -> concatenate shapes; [depth, height, width, in_channels, out_channels]
    weights = tf.get_variable(
        scope+'/conv/weights', shape,
        initializer=tf.truncated_normal_initializer())
    outs = tf.nn.conv3d(
        inputs, weights, strides=(1, 1, 1, 1, 1), padding=padding,
        name=scope+'/conv')
    return tf.contrib.layers.batch_norm(
        outs, decay=0.9, epsilon=1e-5, fused=True, activation_fn=tf.nn.relu,
        updates_collections=None, scope=scope+'/batch_norm', is_training=is_training)


# %% upsample layers

# Note that upsample_layer_NN/FT and upsample_layer are incompatible for 
# saver.restore, since NN/FT blow up dims with all channels, while 
# conv_transpose produces half of channels

@deprecated("this is outdated.  Please just put upsampleNN and conv_layer in sequence")
def upsample_layerNN(inputs, out_num, kernel_size, padding, is_training, scope,
                     upsam_size=(2,2,2), use_batch_renorm=False, 
                     use_batch_norm=False, use_dropout=False,
                     activation="default", init="default", 
                     batch_renorm_scheme=None):
    if activation == "default":
        activation = tf.nn.relu
    if init=="default":
        init = he_init
    outs = upsample_NN(inputs, upsam_size, name=scope+'/upsam_NN')
    outs = tf.layers.conv3d(outs, 
                        out_num, 
                        kernel_size, 
                        strides=(1, 1, 1), 
                        padding=padding,
                        data_format='channels_last', # related to axis down
                        activation=activation,
                        use_bias=True,  # not needed when using batch_norm
                        kernel_initializer=init,
                        bias_initializer=tf.zeros_initializer(),
                        name=scope+'/conv')
    if use_dropout:
        outs = tf.layers.dropout(outs, 
                                 rate=0.1, 
                                 training=is_training, 
                                 name=scope+'/dropout')
    if use_batch_renorm:
        if use_batch_norm:
            warn(scope + ": Both batch_renorm and batch_norm are True. " +
                 "Will use batch_renorm.")
        if not batch_renorm_scheme:
            dict(rmin=0, rmax=np.inf, dmax=np.inf)  # default, no clipping
        outs = tf.layers.batch_normalization(outs,
                        renorm=True,
                        renorm_clipping=batch_renorm_scheme,
                        # default
                        renorm_momentum=0.99,  # default
                        axis = -1, # related to data_format of conv
                        momentum=0.99,  # default
                        epsilon=1e-3,  # default
                        training=is_training,
                        name=scope+'/batch_norm',
                        fused=True)
    elif use_batch_norm:
            #TODO: try regularizing beta and gamma (?)
            outs = tf.layers.batch_normalization(outs, 
                        axis = -1, # related to data_format of conv
                        momentum=0.99,  # default
                        epsilon=1e-3,  # default
                        training=is_training,
                        name=scope+'/batch_norm',
                        fused=True)
    return outs

# lots of code doubled from upsample_layerNN
@deprecated("this is outdated.  Please just put upsampleFT and conv_layer in sequence")
def upsample_layerFT(inputs, out_num, kernel_size, padding, is_training, scope,
                     upsam_size=(2,2,2), use_batch_renorm=False,
                     use_batch_norm=False, use_dropout=False,
                     activation="default", init="default"):
    if activation == "default":
        activation = tf.nn.relu
    if init == "default":
        init = he_init
    outs = upsample_FT(inputs, upsam_size, name=scope+'/upsam_FT')
    outs = tf.layers.conv3d(outs, 
                        out_num, 
                        kernel_size, 
                        strides=(1, 1, 1), 
                        padding=padding,
                        data_format='channels_last', # related to axis down
                        activation=activation,
                        use_bias=True,  # not needed when using batch_norm
                        kernel_initializer=init,
                        bias_initializer=tf.zeros_initializer(),
                        name=scope+'/conv')
    if use_dropout:
        outs = tf.layers.dropout(outs, 
                                 rate=0.1, 
                                 training=is_training, 
                                 name=scope+'/dropout')
    if use_batch_renorm:
        if use_batch_norm:
            warn(scope + ": Both batch_renorm and batch_norm are True. " +
                 "Will use batch_renorm.")
        outs = tf.layers.batch_normalization(outs,
                        renorm=True,
                        renorm_clipping=dict(rmax=np.inf, rmin=0, dmax=np.inf),
                        # default
                        renorm_momentum=0.99,  # default
                        axis = -1, # related to data_format of conv
                        momentum=0.99,  # default
                        epsilon=1e-3,  # default
                        training=is_training,
                        name=scope+'/batch_norm',
                        fused=True)
    elif use_batch_norm:
            #TODO: try regularizing beta and gamma (?)
            outs = tf.layers.batch_normalization(outs, 
                        axis = -1, # related to data_format of conv
                        momentum=0.99,  # default
                        epsilon=1e-3,  # default
                        training=is_training,
                        name=scope+'/batch_norm',
                        fused=True)
    return outs

@deprecated("the recommended function is upsample_layerNN. If strided upconv " + 
            "is desired, use upsample_layer.")
def upsample_layer_original(inputs, out_num, kernel_size, padding, scope, 
                            is_training): 
    shape = list(kernel_size) + [out_num, inputs.shape[-1].value]
    input_shape = inputs.shape.as_list()
    out_shape = [input_shape[0]] + \
        list(map(lambda x: x*2, input_shape[1:-1])) + [out_num]
    weights = tf.get_variable(
        scope+'/deconv/weights', shape,
        initializer=tf.truncated_normal_initializer())
    outs = tf.nn.conv3d_transpose(
        inputs, weights, out_shape, (1, 2, 2, 2, 1), padding=padding, name=scope+'/deconv')
    return tf.contrib.layers.batch_norm(
        outs, decay=0.9, epsilon=1e-5, fused=True, activation_fn=tf.nn.relu,
        updates_collections=None, scope=scope+'/batch_norm', is_training=is_training)


# %% old helper functions for upsampling 
###### all these work, but the above is preferred
#def tf_repeat3d_batch(t, rd, rh, rw):  # data_format='channels_last', name='NN_upsampling'
#    """
#    calls tf_repeat_nd to achieve NN upsampling.
#    t is a tf-tensor with shape (n_batches, depth, height, width, channels)  
#    ("channels last!")
#    rd, rh and rw are repeats in depth, height and width respectively.
#    """
#    # Problem:  this loses shape info for some reason
#    return tf_repeat_nd(t, [1, rd, rh, rw, 1])
#
#
## from github issue
#def tf_repeat_nd(tensor, repeats):
#    """
#    Args:
#
#    input: A Tensor. 1-D or higher.
#    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input
#
#    Returns:
#    
#    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
#    """
#    with tf.variable_scope("repeat"):
#        expanded_tensor = tf.expand_dims(tensor, -1)
#        multiples = [1] + repeats
#        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
#        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
#    # Problem:  this loses shape info for some reason
#    return repeated_tensor
#
#
#@deprecated("the recommended function is tf_repeat3d_batch")
#def tf_repeat3d_batch_original(t, rd, rh, rw):  # data_format='channels_last', name='NN_upsampling'
#    """
#    calls tf_repeat_along_axis5d 3 times to achieve NN upsampling.
#    t is a tf-tensor with shape (n_batches, depth, height, width, channels)  
#    ("channels last!")
#    rd, rh and rw are repeats in depth, height and width respectively.
#    """
#    return tf_repeat_along_axis5d(
#                tf_repeat_along_axis5d(
#                        tf_repeat_along_axis5d(t, rd, 1), rh, 2), rw, 3)
#
#@deprecated("the recommended function is tf_repeat_nd")
#def tf_repeat_along_axis5d(t, repeats, axis):
#    """designed to imitatate np.repeat(t, repeats, axis=axis) for 5-dim t"""
#    if axis == 0:
#        return tf.transpose(tf_repeat_along_axis5d(tf.transpose(t), repeats, 4)) # v
#    elif axis == 1:
#        return tf.transpose(tf_repeat_along_axis5d(tf.transpose(t, (2,3,4,0,1)), repeats, 4), (3,4,0,1,2))
#    elif axis == 2:
#        return tf.transpose(tf_repeat_along_axis5d(tf.transpose(t, (3,4,0,1,2)), repeats, 4), (2,3,4,0,1))
#    elif axis == 3:
#        return tf.transpose(tf_repeat_along_axis5d(tf.transpose(t, (4,0,1,2,3)), repeats, 4), (1,2,3,4,0))  # v
#    elif axis == 4:
#        return tf.reshape(tf.tile(tf.reshape(t, (-1, 1)), (1, repeats)), 
#                          (t.get_shape()[0], t.get_shape()[1], 
#                           t.get_shape()[2], t.get_shape()[3], -1)) # v
#    else:
#        msg = ("axis " + str(axis) + " is out of bounds for array of "  + 
#               "dimension " + str(len(t.get_shape())))
#        raise IndexError(msg)
#####

# %% 

#def bias_variable(shape):
#    initial = tf.zeros(shape)
#    return tf.Variable(initial)
#
#def weight_variable(shape):
#    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
#    return tf.Variable(initial)