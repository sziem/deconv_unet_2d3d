#import tensorflow as tf
import numpy as np
from warnings import warn

# own code
from .decorators import deprecated

# for tensorflow 1.14 use this to avoid warnings:
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

"""
This module provides conv-layers, pool-layers, upsampling-layers.
Conv Layers contain convolution, activation, and possibly dropout or batch_norm
"""
# TODO: allow kwargs to set hyperparams of batch norm etc.
# TODO: use channels_last or channels_first depending on GPU/CPU automatically
# (should be in a higher abstraction though); see resnet-code
# sth. like tf.compiled_with_cuda()

# %% conv layers
# 'v3'
# (preprocessing)  -> 
# input-layer:                                 conv (no bias) -> relu
# conv-layer: (dropout) -> (batch_(re)norm) -> conv (no bias) -> relu
# ouput-layer:             (batch_(re)norm) -> conv (no bias)
# -> (postprocessing)

# some adaptations needed if I want to do resnet

# see MiloMinderinder's answer at 
# https://stackoverflow.com/questions/39691902
# (don't know if this is still up to date)

def input_layer_v3(
        inputs, out_num, conv_size, padding, is_training, scope, activation, 
        init, data_format='channels_last'):
    # transforms layer to first feature representation
    # conv + bias -> relu
    # TODO: try no activation in input.
    # is_training=False param does not do anything here
    # no dropout and batch(re)norm
    return conv_layer_v3(inputs=inputs, 
                         out_num=out_num, 
                         conv_size=conv_size, 
                         padding=padding, 
                         is_training=is_training,  # does not matter
                         scope=scope,
                         use_batch_renorm=False, 
                         use_batch_norm=False, 
                         use_dropout=False, 
                         activation=activation, 
                         init=init, 
                         batch_renorm_scheme=None,  # does not matter
                         data_format=data_format)
    
def output_layer_v3(
        inputs, out_num, conv_size, padding, is_training, scope, init, 
        use_batch_renorm=False, use_batch_norm=False, batch_renorm_scheme=None, 
        data_format='channels_last'):
    # "classification" layer --> transforms features into output image
    # no dropout 
    # batch(re)nnorm -> conv (no bias)
    # TODO: try activating somehow?
    return conv_layer_v3(inputs=inputs, 
                         out_num=out_num, 
                         conv_size=conv_size, 
                         padding=padding, 
                         is_training=is_training,
                         scope=scope,
                         activation=None, 
                         init=init, 
                         use_batch_renorm=use_batch_renorm, 
                         use_batch_norm=use_batch_norm, 
                         use_dropout=False, 
                         batch_renorm_scheme=batch_renorm_scheme,
                         data_format=data_format)

def conv_layer_v3(
        inputs, out_num, conv_size, padding, is_training, scope, activation, 
        init, use_batch_renorm=False, use_batch_norm=False, use_dropout=False, 
        dropout_rate=0.05, batch_renorm_scheme=None, data_format='channels_last'):
    # conv_size is conv_kernel_size
    # bn->conv->relu
    
    # TODO allow kwargs to set these:
    # do kwargs: rate
    # bn kwargs: momentum, epsilon
    # brn kwargs: momentum, epsilon, renorm_momentum, renorm_clipping
    
    if len(conv_size) == len(inputs.shape) - 2: # minus image_dim and channel_dim
        ndims = len(conv_size)
    else:
        raise ValueError(
                "conv_size must have len(inputs.shape) - 2 entries. " +
                "conv_size is " + str(conv_size) + " and inputs has shape " +
                str(inputs.shape))

    # parsing input
    if data_format == 'channels_last' or data_format == 'NDHWC':  # recommended for CPU
        axis = -1 
    elif data_format == 'channels_first' or data_format == 'NCDHW':  # recommended for GPU
        axis = 1
        
    if not (use_batch_norm or use_batch_renorm):
        use_bias = True
    else:
        use_bias = False
    
    outs = inputs
    # TODO: not sure where it is customary to put this.
    if use_dropout:  # dropout + batch_norm = :(
        outs = tf.layers.dropout(outs, 
                                 rate=dropout_rate, 
                                 training=is_training, 
                                 name=scope+'/dropout')

    # I think batch_norm could just be replaced by subtracting population mean
    # or any number suitable to the scale of activations to make input to 
    # conv layer not all positive
    if use_batch_renorm:  
        if use_batch_norm:
            warn(scope + ": Both batch_renorm and batch_norm are True. " +
                 "Will use batch_renorm.")
#        if not batch_renorm_scheme:
            # disabling this.  Just use default as is
#            dict(rmin=0, rmax=np.inf, dmax=np.inf)  # default, no clipping
        outs = tf.layers.batch_normalization(outs,
                        renorm=True,
                        renorm_clipping=batch_renorm_scheme,
                        renorm_momentum=0.95,  # default: 0.99
                        axis=axis, # related to data_format of conv
                        momentum=0.99,  # default
                        epsilon=1e-3,  # default 1e-3
                        training=is_training,
                        name=scope+'/batch_norm',
                        fused=True)
    elif use_batch_norm:
            outs = tf.layers.batch_normalization(outs, 
                        axis=axis, # related to data_format of conv
                        momentum=0.99,  # default
                        epsilon=1e-3,  # default
                        training=is_training,
                        name=scope+'/batch_norm',
                        fused=True)        
    
    strides = ndims*(1,)
    if ndims == 2:
        outs = tf.layers.conv2d(
                outs, 
                out_num, 
                conv_size, 
                strides=strides, 
                padding=padding,
                data_format=data_format,
                activation=activation,
                use_bias=use_bias,  # not needed whens using batch_norm
                kernel_initializer=init,
                bias_initializer=tf.zeros_initializer(),
                name=scope+'/conv_relu')
    elif ndims == 3:
        outs = tf.layers.conv3d(
                outs, 
                out_num, 
                conv_size, 
                strides=strides, 
                padding=padding,
                data_format=data_format,
                activation=activation,
                use_bias=use_bias,  # not needed whens using batch_norm
                kernel_initializer=init,
                bias_initializer=tf.zeros_initializer(),
                name=scope+'/conv_relu')
    else:
        raise RuntimeError(
                "conv_layer_v3 is only implemented for 2d or 3d images, " +
                "i.e. 4d or 5d net input. ndims was detected as " + str(ndims))
    return outs

# %% downsampling via pooling

# TODO: code doubling.  Better to outsource checks to build_network or similar
def max_pool(inputs, pool_size, padding, scope, data_format='channels_last'):
    if len(pool_size) == len(inputs.shape) - 2: # minus image_dim and channel_dim
        ndims = len(pool_size)
    else:
        raise ValueError(
                "pool_size must have len(inputs.shape)-2 entries. " +
                "pool_size is " + str(pool_size) + " and inputs has shape " +
                str(inputs.shape))
    if ndims == 2:
        return tf.layers.max_pooling2d(inputs, 
                                       pool_size, 
                                       pool_size, 
                                       padding=padding,
                                       data_format=data_format,
                                       name=scope+'/pool')
    elif ndims == 3:
        return tf.layers.max_pooling3d(inputs, 
                                       pool_size, 
                                       pool_size, 
                                       padding=padding,
                                       data_format = data_format,
                                       name=scope+'/pool')
    else:
        raise RuntimeError(
                "max_pool is only implemented for 2d or 3d images, " +
                "i.e. 4d or 5d net input. ndims was detected as " + str(ndims))


def avg_pool(inputs, pool_size, padding, scope, data_format='channels_last'):
    if len(pool_size) == len(inputs.shape) - 2: # minus image_dim and channel_dim
        ndims = len(pool_size)
    else:
        raise ValueError(
                "pool_size must have len(inputs.shape)-2 entries. " +
                "pool_size is " + str(pool_size) + " and inputs has shape " +
                str(inputs.shape))
    if ndims == 2:
        return tf.layers.average_pooling2d(inputs, 
                                           pool_size, 
                                           pool_size, 
                                          padding=padding,
                                          data_format=data_format,
                                          name=scope+'/pool')
    elif ndims == 3:
        return tf.layers.average_pooling3d(inputs, 
                                           pool_size, 
                                           pool_size, 
                                           padding=padding,
                                           data_format = data_format,
                                           name=scope+'/pool')
    else:
        raise RuntimeError(
                "avg_pool is only implemented for 2d or 3d images, " +
                "i.e. 4d or 5d net input. ndims was detected as " + str(ndims))


# %% custom upsampling layers (in 3d)

def upsample_NN(inputs, upsam_size, scope, data_format='channels_last'):
    """
    upsample t in spatial dimensions using Nearest Neighbour upsampling
    """
    if data_format == "channels_last" or data_format == "NDHWC":
        # first [1] for n_images (no upsampling), 
        # then repeats in d, h, w 
        # and finally [1] for channels (no upsampling)
        multiples = [1] + list(upsam_size) + [1]
    elif data_format == "channels_first" or data_format == "NCDHW":
        warn("upsampling with channels_first has not been tested.")
        # first [1] for n_images (no upsampling), 
        # then [1] for channels (no upsampling) 
        # and finally repeats in d, h, w  
        multiples = [1] + [1] + list(upsam_size)
    with tf.name_scope(scope):
        return _nearest_neighbor_upsampling(inputs, multiples)


# upsampling by padding with zero-frequencies
# use window fxn to avoid sharp border?
def upsample_FT(inputs, upsam_size, scope, data_format='channels_last'):
    
    if data_format == 'channels_first' or data_format == 'NCDHW':
        raise RuntimeError("This has not been tested for channels_first")
    sh = np.array(inputs.shape.as_list())
    f_ny_old = sh//2  #nyqvist frequency of original tensor

    with tf.name_scope(scope):
        t_cmplx = tf.complex(inputs, tf.zeros(inputs.shape))
        t_cmplx_ft = tf.fft3d(t_cmplx)
        t_cmplx_ft_pad = tf.manip.roll(t_cmplx_ft, f_ny_old, axis=(0,1,2))
        t_cmplx_ft_pad = tf.pad(t_cmplx_ft_pad,
                            ((0, (upsam_size[0]-1) * t_cmplx_ft.shape[0]), 
                             (0, (upsam_size[1]-1) * t_cmplx_ft.shape[1]), 
                             (0, (upsam_size[2]-1) * t_cmplx_ft.shape[2])),
                            'constant')
        t_cmplx_ft_pad = tf.manip.roll(t_cmplx_ft_pad, -f_ny_old, axis=(0,1,2))
        t_upsam = tf.real(tf.ifft3d(t_cmplx_ft_pad))
    # the test found a significant imag part though --> bc of hard edge?
    return t_upsam


# %% utils for resnet-blocks:
# a.k.a "projection_shortcuts"

def upsample_features_NN(inputs, feature_upsam, name=None, data_format='channels_last'):
    """feature_upsam is a scalar"""
    if data_format == "channels_last" or data_format == "NDHWC":
        # first [1] for n_images (no upsampling), 
        # then ones in (d,) h, w  (no upsampling)
        # and finally [features_upsm] for channels 
        multiples = [1] + list(np.ones_like(inputs.shape)[1:-1]) + [feature_upsam]
    elif data_format == "channels_first" or data_format == "NCDHW":
        warn("upsampling with channels_first has not been tested.")
        # first [1] for n_images (no upsampling),
        # then [features_upsm] for channels 
        # and finally ones in (d,) h, w  (no upsampling)
        multiples = [1] + [feature_upsam] + list(np.ones_like(inputs.shape)[1:-1])
    with tf.name_scope(name):
        return _nearest_neighbor_upsampling(inputs, multiples)

@deprecated("this works, but I'd recommend upsample_features_NN")
# TODO: think about arch
def upsample_features_conv(inputs, feature_upsam, scope=None, init="default", 
                           data_format='channels_last'):
    """
    - no batch_(re)norm and no dropout (maybe I should add these)
    - 1x1-conv increasing number of features by feature_upsam, 
    - no actiavtion
    - no bias
    """
    if init == "default":
        init = he_init
    if data_format == 'channels_last' or data_format == 'NDHWC':  # recommended for CPU
        c_axis = -1 
    elif data_format == 'channels_first' or data_format == 'NCDHW':  # recommended for GPU
        c_axis = 1
    return tf.layers.conv3d(inputs, 
                            feature_upsam*inputs[c_axis],
                            kernel_size=(1,1,1),
                            strides=(1,1,1),
                            padding='same', # should not make a difference (?)
                            data_format=data_format,
                            activation=None,
                            use_bias=False,
                            kernel_initializer=init,
                            name=scope+'/conv1x1')

# %% helper for upsampling

# should work for arbitrary ndims>1
def _nearest_neighbor_upsampling(t, multiples):
    """
    Args:
        t: tensor with rank >=1
        multiples: list(!) of repeats in every dim
        
    The following conditions must be satisfied:
        len(multiples) = len(t.shape)
        multiples can only be a scalar (not a list) in every dimension

    from: https://github.com/tensorflow/tensorflow/issues/8246

    Example:

    this is an attempt to implement np.repeat in tf.
    NOTE THAT THIS DOES NOT PROVIDE ALL THE FUNCTIONALITY OF NP-REPEAT!
    FOR EXAMPLE, THIS WILL NOT WORK:
    x = np.array([0,1,2])
    np.repeat(x,[3,4,5])
    >>> array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    
    A positive example is:
    nearest_neighbor_upsampling(t, [2,2])   (Run within session)
    
    This is a replacement for
    x = np.arange(4).reshape(2,2)
    np.repeat(x, [2,2])
    >>> array([[0, 0, 1, 1], 
               [0, 0, 1, 1],
               [2, 2, 3, 3],
               [2, 2, 3, 3])
    """
    if np.prod(np.shape(multiples)) != len(t.shape):
        raise ValueError("multiples is not a list of correct shape.")
    not_none = np.not_equal(np.array(t.shape.as_list()), None) # one dim might have undeterm. shape
    new_shape = np.array(t.shape)
    new_shape[not_none] = new_shape[not_none]*np.array(multiples)[not_none]
    new_shape[np.logical_not(not_none)] = -1  # infer when reshaping
    
    # I found this a bit complicated to think through, but it works
    expanded_tensor = tf.expand_dims(t, -1) # because this works ...
    multiples = [1] + multiples             # because this works ...
    tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
    
    return tf.reshape(tiled_tensor, new_shape)

# %% outdated stuff

#def upsample_NN_orig(t, upsam_size=(2,2,2), name=None, data_format='channels_last'):
#    """
#    modified tf_repeat_nd to achieve NN upsampling keeping shape info.
#    t is a tf-tensor with shape (n_batches, depth, height, width, channels)  
#    ("channels last!")
#    """
#    if data_format == "channels_last" or "NDHWC":
#        # first [1] for broadcasting, then [1] for n_images (no upsampling),
#        # then repeats in d, h, w and finally [1] for channels (no upsampling)
#        multiples = [1] + [1] + list(upsam_size) + [1]
#        reshape_size = (-1, # infer; but shouldn't it simply be t.shape[0]?
#                        upsam_size[0]*t.shape[1], # d_dim
#                        upsam_size[1]*t.shape[2], # h_dim
#                        upsam_size[2]*t.shape[3], # w_dim
#                        t.shape[4]) # c_dim
#    elif data_format == "channels_first" or "NCDHW":
#        warn("upsampling with channels_first has not been tested.")
#        # first [1] for broadcasting, then [1] for n_images (no upsampling),
#        # then [1] for channels (no upsampling) and finally repeats in d, h, w  
#        multiples = [1] + [1] + [1] + list(upsam_size)
#        reshape_size = (-1, # infer; but shouldn't it simply be t.shape[0]?
#                        t.shape[1], # c_dim
#                        upsam_size[0]*t.shape[2], # d_dim
#                        upsam_size[1]*t.shape[3], # h_dim
#                        upsam_size[2]*t.shape[4]) # w_dim
#    with tf.name_scope(name):
#        expanded_tensor = tf.expand_dims(t, -1)
#        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
#        t_up = tf.reshape(tiled_tensor, reshape_size)
#    return t_up