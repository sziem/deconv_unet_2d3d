#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:34:45 2018

@author: soenke
"""

# TODO: utils should be network-architecture-independent, but currently that
# is not the case for all utils

# TODO: update docstrings
# TODO: extensive testing!
# TODO: organize this well; add postprocessing methods
# TODO: adjust all ops so that they work on last 4 dims only and document this
# such that first dim can be None
# TODO: enable working on tf.shape-objects or tuples instead of lists

import tensorflow as tf
import numpy as np
from warnings import warn

# %% general stuff  -->  move to tf_toolbox
def crop(t, new_shape):
    """wrapper for crop_middle.  See that for docs."""
    return crop_middle(t, new_shape)

# check tf.image.central_crop -> only 2d
def crop_middle(t, new_shape):
    """
    will crop half the pixels on the left and half on the right of t st. it
    has new_shape. If the difference in shapes is odd in some dim, it will 
    crop the extra pixel on the right/bottom/... .
    
    Works for n-dimensional tensors
    
    Ignores any dimensions that is None in either new_shape or t.
    """
    input_shape = convert_shape_to_np_array(t.shape)
    new_shape = convert_shape_to_np_array(new_shape)

    # detect Nones
    is_none_in = input_shape == np.array(None)
    is_none_new = new_shape == np.array(None)
    is_none = is_none_in + is_none_new
    
    not_none = np.logical_not(is_none)
    if len(input_shape) != len(new_shape):
        raise ValueError("new_shape must have same number of dims as t.\n" +
                         "len(t.shape) is " + str(len(t.shape)) + ".\n" +
                         "len(new_shape) is " + str(len(new_shape)) + ".")
    if np.any(new_shape[not_none] > input_shape[not_none]):
        raise ValueError("All entries of new_shape must be smaller than t.shape.\n" + 
                         "t.shape is " + str(input_shape) + ".\n" +
                         "new_shape is " + str(new_shape) + ".")
    if np.any(new_shape[not_none] < 0):
        raise ValueError("All entries of new_shape must be greater than 0.\n" +
                         "new_shape is " + str(new_shape) + ".")
    
    # slices are created here
    starts = np.zeros_like(input_shape, dtype=object) # dtype in case of None
    ends = np.zeros_like(starts)
    for i, ignore in enumerate(is_none):
        if not ignore:
            starts[i] = (input_shape[i]-new_shape[i])//2
            ends[i] = starts[i] + new_shape[i]
        else:
            starts[i] = 0
            ends[i] = None

    slices = tuple(map(slice, starts, ends))
    with tf.name_scope("crop"):
        return t[slices]


def crop_right(t, new_shape):
    """
    will crop pixels on the right of t st. it has new_shape. 
    There was a TODO-comment to implement this, but I don't remember what for.
    Usually prefer central_crop
    
    Works for n-dimensional tensors
    
    Ignores any dimensions that is None in either new_shape or t.
    """
    input_shape = convert_shape_to_np_array(t.shape)
    new_shape = convert_shape_to_np_array(new_shape)

    # detect None
    is_none_in = input_shape == np.array(None)
    is_none_new = new_shape == np.array(None)
    is_none = is_none_in + is_none_new
    
    not_none = np.logical_not(is_none)
    if len(input_shape) != len(new_shape):
        raise ValueError("new_shape must have same number of dims as t.\n" +
                         "len(t.shape) is " + str(len(t.shape)) + ".\n" +
                         "len(new_shape) is " + str(len(new_shape)) + ".")
    if np.any(new_shape[not_none] > input_shape[not_none]):
        raise ValueError("All entries of new_shape must be smaller than t.shape.\n" + 
                         "t.shape is " + str(input_shape) + ".\n" +
                         "new_shape is " + str(new_shape) + ".")
    if np.any(new_shape[not_none] < 0):
        raise ValueError("All entries of new_shape must be greater than 0.\n" +
                         "new_shape is " + str(new_shape) + ".")
    
    # slices are created here
    starts = np.zeros_like(input_shape, dtype=object) # dtype in case of None
    ends = np.zeros_like(starts)
    for i, ignore in enumerate(is_none):
        if not ignore:
            starts[i] = 0
            ends[i] = starts[i] + new_shape[i]
        else:
            starts[i] = None
            ends[i] = None

    slices = tuple(map(slice, starts, ends))
    with tf.name_scope("crop"):
        return t[slices]    

def pad_right(t, new_shape, padding_value=0):
    """
    will pad t on the right with zeroes until it has new_shape.
    
    ignores any dimensions that have been set to None in either t or new_shape
    """
    input_shape = convert_shape_to_np_array(t.shape)
    new_shape = convert_shape_to_np_array(new_shape)

    # detect None
    is_none_in = input_shape == np.array(None)
    is_none_new = new_shape == np.array(None)
    is_none = is_none_in + is_none_new
    not_none = np.logical_not(is_none)
    
    n_pad = list()  # contains paddings on [left, right]
    pixel_diffs = np.zeros_like(input_shape)
    for i, nn in enumerate(not_none):
        if nn:
            pixel_diffs[i] = new_shape[i] - input_shape[i]
        n_pad.append([0, pixel_diffs[i]])

    if np.any(pixel_diffs < 0):
        raise ValueError("new_shape must be greater or equal input shape.")
    
    return tf.pad(t, n_pad, mode='CONSTANT', constant_values=padding_value)


# %% prepare input into loss function

def pretend_network_pass_slice(t, network_depth, padding, conv_size=(3,3,3), 
                         pool_size=(2,2,2)):
    """pretend t was propagated through network.  Will change shape of t, if
       necessary."""
    # TODO: squeeze and expand outside of utils and use just 1 fxn!
    # TODO: maybe it would be nicer to put preprocess_input_shape here?
    t = tf.expand_dims(t,0)  # because slice is missing batch_dimension
    if padding == "valid":
        t = _crop_to_valid_net_output_shape(t, network_depth, 
                conv_size=conv_size, pool_size=pool_size)
    # Do nothing for same padding.
    return tf.squeeze(t,0)

def pretend_network_pass(t, network_depth, padding, conv_size=(3,3,3), 
                         pool_size=(2,2,2)):
    """pretend t was propagated through network.  Will change shape of t, if
       necessary."""
    # TODO: maybe it would be nicer to put preprocess_input_shape here?
    if padding == "valid":
        t = _crop_to_valid_net_output_shape(t, network_depth, 
                conv_size=conv_size, pool_size=pool_size)
    # Do nothing for same padding.
    return t

def _crop_to_valid_net_output_shape(t, network_depth, conv_size=(3,3,3), 
                                pool_size=(2,2,2)):
    """
    This removes all pixels that were affected by padding from output.
    It is meant to be used for inputting y into the loss function for valid
    padding or for same padding with cut_loss_to_valid.
    
    Note that output shape might be smaller than (shape - n_pixels_lost), since 
    input is first cropped to valid unet input.
    """
    with tf.name_scope("crop_to_valid"):
        t_crop = preprocess_input_shape(t, network_depth, "valid",
                conv_size=conv_size, pool_size=pool_size)
        out_shape = get_net_output_shape(t_crop.shape, network_depth, 
                "valid", conv_size=conv_size, pool_size=pool_size)
        t_crop = crop(t_crop, out_shape)
    return t_crop


# %% Tools to crop skips for concatenation using valid padding

#def fix_skip_size(skip_connect, padding, block_output_shapes):
#    """for valid padding, the size of the skip-connections must be adapted"""
#    if padding == "valid":
##        if block_index == 0:
##            sequence = ("conv", "conv", "conv")
##        else:
##            sequence = ("conv", "conv")
#        concat_shape = block_output_shapes.pop()#[-(block_index+1)]
#        concat_shape = get_layer_sequence_input_shape(concat_shape,
#                sequence, concat_shape[-1], padding, conv_size=(3,3,3))
#        skip_connect = crop(skip_connect, concat_shape)
#    # do nothing for "same" padding
#    return skip_connect


#%% Tools to prepare unet input (preprocessing)

## TODO make utils ready for slice input (4d w/o batch dimension)
#def preprocess_input_shape_slice(t, network_depth, padding, conv_size=(3,3,3),
#                                 pool_size=(2,2,2)):
#    t = tf.expand_dims(t,0)  # because slice is missing batch_dimension  #DBG?
#    if is_allowed_input_shape(t.shape, network_depth, padding,
#                              conv_size=conv_size, pool_size=pool_size):
#        return tf.squeeze(t,0)
#    elif padding == "same":
#        print("padding with zeroes on the right to nearest allowed " +
#              "input image size.")
#        t = _pad_to_nearest_allowed_input_shape(t, network_depth, padding, 
#                            conv_size=conv_size, pool_size=pool_size)
#    elif padding == "valid":
#        # TODO: move print out of here so that it is not called during pretend_network_pass
#        print("Cropping to nearest allowed input image size.")
#        t = _crop_to_nearest_allowed_input_shape(t, network_depth, padding, 
#                            conv_size=conv_size, pool_size=pool_size)
#    return tf.squeeze(t,0)
#
#def preprocess_input_shape(t, network_depth, padding, conv_size=(3,3,3),
#                           pool_size=(2,2,2)):
#    if is_allowed_input_shape(t.shape, network_depth, padding,
#                              conv_size=conv_size, pool_size=pool_size):
#        return t
#    elif padding == "same":
#        print("padding with zeroes on the right to nearest allowed " +
#              "input image size.")
#        t = _pad_to_nearest_allowed_input_shape(t, network_depth, padding, 
#                            conv_size=conv_size, pool_size=pool_size)
#    elif padding == "valid":
#        # TODO: move print out of here so that it is not called during pretend_network_pass
#        print("Cropping to nearest allowed input image size.")
#        t = _crop_to_nearest_allowed_input_shape(t, network_depth, padding, 
#                            conv_size=conv_size, pool_size=pool_size)
#    return t

#def _pad_to_nearest_allowed_input_shape(t, network_depth, padding,
#                                         conv_size=(3,3,3), pool_size=(2,2,2)):
#    """
#    Pad t on the right/lower/back border with zeroes to make it have an 
#    allowed input shape.  This is usually used to prepare input to a net with 
#    'same' padding.
#    """
#    # old implementation for 'same'
#    # left this here because it shows in which layer how may zeroes are added.
##    spatial_shape = input_shape[1:4]
##    n_zeroes_for_padding = [[0,0], [0,0], [0,0], [0,0], [0,0]]  # [left, right]
##    for j in range(3):  # z, y, x
##        for i in range(network_depth-1):
##            if spatial_shape[j] % 2 == 1:
##                n_zeroes_for_padding[j+1][1] += 2**i  # z,y,x
##            spatial_shape[j] = int(np.ceil(spatial_shape[j]/2))
##    print("padding with zeroes [left, right]:", n_zeroes_for_padding, 
##          "to nearest allowed image size.")
#    new_shape = _get_nearest_larger_allowed_input_shape(t.shape, 
#            network_depth, padding, conv_size=conv_size, pool_size=pool_size)
#    return pad_right(t, new_shape)
#
#def _crop_to_nearest_allowed_input_shape(t, network_depth, padding, 
#                                         conv_size=(3,3,3), pool_size=(2,2,2)):
#    """
#    remove pixels on the border to make input have an allowed input shape.
#    This is usually used to prepare input to a net with 'valid' padding.
#    """
#    new_shape = _get_nearest_smaller_allowed_input_shape(t.shape, 
#            network_depth, padding, conv_size=conv_size, pool_size=pool_size)
#    return crop(t, new_shape)
#
#def _get_nearest_larger_allowed_input_shape(in_shape, network_depth, padding,
#                                        conv_size=(3,3,3), pool_size=(2,2,2)):
#    # TODO: for case where in_shape << allowed_shape, this will be very inefficient.
#    # But then this case should rarely arise.
#    in_shape = convert_shape_to_np_array(in_shape)
#    new_shape = (in_shape[0],)  # n_images axis
#    max_pixels = np.max(in_shape[1:4]) + 2**(network_depth-1)
#    allowed_sizes = allowed_n_input_pixels(network_depth, padding,
#        conv_size=conv_size, pool_size=pool_size, max_pixels=max_pixels)
#    for dim, size in enumerate(in_shape[1:4]):
#        idx = np.searchsorted(allowed_sizes, size, side="left")
#        while not idx < len(allowed_sizes):
#            warn("in_shape is much smaller than allowed in_shape. " +
#                 "Input will be padded a lot.")
#            allowed_sizes = allowed_n_input_pixels(network_depth, padding,
#                conv_size=conv_size, pool_size=pool_size, 
#                max_pixels=2*max_pixels)  # arbitrarily doubling.
#        new_shape += (allowed_sizes[idx],)  # spatial axes
#    new_shape += (in_shape[-1],)  # channel axis
#    return new_shape
#
#def _get_nearest_smaller_allowed_input_shape(in_shape, network_depth, padding,
#                                        conv_size=(3,3,3), pool_size=(2,2,2)):
#    in_shape = convert_shape_to_np_array(in_shape)
#    new_shape = (in_shape[0],)  # n_images axis
#    max_pixels = np.max(in_shape[1:4])
#    allowed_sizes = allowed_n_input_pixels(network_depth, padding,
#            conv_size=conv_size, pool_size=pool_size, max_pixels=max_pixels)
#    if not allowed_sizes:
#        raise RuntimeError("No shape smaller or equal the provided input " +
#                           "shape is allowed. Consider larger input images " + 
#                           "or use a smaller network_size.")
#    for dim, size in enumerate(in_shape[1:4]):
#        idx = np.searchsorted(allowed_sizes, size, side="left")
#        new_shape += (allowed_sizes[idx-1],)
#    new_shape += (in_shape[-1],)  # channel axis
#    return new_shape


#%% check if input shape is allowed for unet 
#def is_allowed_input_shape(in_shape, network_depth, padding, conv_size=(3,3,3), 
#                           pool_size=(2,2,2)):
#    """
#    determine if input shape is allowed for input to the unet with given 
#    parameters.  Not all input shapes are allowed since it must be possible
#    to concatenate up-path and down-path.
#    """
#    # ignores batch shape
#    in_shape = convert_shape_to_np_array(in_shape)
#    is_allowed = in_shape[-1] == 1
#    if not is_allowed: 
#        raise ValueError("Last dimension of input (n_channels) should be 1.")
#    max_pixels = np.max(in_shape[1:4])
#    allowed_pixels = allowed_n_input_pixels(network_depth, padding,
#        conv_size=conv_size, pool_size=pool_size, max_pixels=max_pixels)
#    #false_dims = list()
#    for n_pixels in in_shape[1:4]:
#        is_allowed = is_allowed and (n_pixels in allowed_pixels)
#    return is_allowed
        

#def allowed_n_input_pixels(network_depth, padding, conv_size=(3,3,3), 
#                        pool_size=(2,2,2), max_pixels=4096):
#    """
#    return a list of allowed number of input pixels in each spatial dimension
#    so that concatenation from down and up path is possible.
#    max_pixels is an (arbitrary) upper limit for the contents of the list.
#    
#    For example, if the list is [4,8,12], a 4x12x8-image would be acceptable
#    input to the unet.
#    
#    Calculates allowed input by starting with the smallest possible output
#    of the bottom layer and calculating the input shapes of the previous layers 
#    up to the first.
#    """    
#    # actually, allowed input shapes to 'same' are n*2**i for any n
#    # and this is a bit overkill, 
#    # but for 'valid', this is not so clear to me.
#
#    # max_pixels is just an arbitrary stopping point
#    batch_size = None # arbitrary, not needed except to have 5d-shape
#    n_channels_bottom = 128 # arbitrary, not needed except to have 5d-shape
#  
#    if padding == "same":        
#        # bottom layer output can be any number >= 1
#        bottom_block_output_pixels = 1
#    elif padding == "valid":
#        # bottom_layer_output must be even st. output from all down layers is even
#        # and > 4 st. network can expand in up-path.  Thus 6 is the minimum possible.
#        bottom_block_output_pixels = 6
#        allowed_input_pixels = []
#
#    # make shape artificially 5d...
#    bottom_block_output_shape = list(
#            (batch_size,) + 3*(bottom_block_output_pixels,) + (n_channels_bottom,))    
#    allowed_input = _get_net_input_shape_from_bottom_block_output_shape(
#            bottom_block_output_shape, padding, network_depth, conv_size, 
#            pool_size)[1]  # ... and slice any of the spatial dimensions.
#    
#    if allowed_input > max_pixels:
#        warn("allowed_input_sizes is empty, since minimum allowed input " +
#             "size " + str(allowed_input) + " is larger than the " + 
#             "specified max_pixels " + str(max_pixels) + ".")
#    
#    allowed_input_pixels = list()
#    while allowed_input <= max_pixels:
#        allowed_input_pixels.append(allowed_input)
#        if padding == "same":
#            bottom_block_output_pixels += 1
#        elif padding == "valid":
#             bottom_block_output_pixels += 2
#        bottom_block_output_shape = list(
#            (batch_size,) + 3*(bottom_block_output_pixels,) + (n_channels_bottom,))
#        allowed_input = _get_net_input_shape_from_bottom_block_output_shape(
#            bottom_block_output_shape, padding, network_depth, conv_size, 
#            pool_size)[1]
#    return allowed_input_pixels


#def _get_net_input_shape_from_bottom_block_output_shape(bottom_block_out_shape,
#        padding, network_depth, conv_size=(3,3,3), pool_size=(2,2,2)):
#    """
#    Determine the input shape from the shape of the output of the bottom layer.
#    This function is used to calculate shapes that can be used as input for a 
#    unet.
#    """
#    used_correct_pool_inputs = True # since am starting from bottom block
#    # sequence of entire down path including bottom_block
#    blocks = ("first_block",)  
#    for layer_index in range(1, network_depth-1):
#        blocks += ("down_block",)
#    blocks += ("bottom_block",)
#    n_channels_start = 32 # arbitrary.  Not used by the function.
#    # TODO make function call cleaner by removing this arg
#    return get_block_sequence_input_shape(bottom_block_out_shape, blocks, 
#            n_channels_start, padding, conv_size=conv_size, pool_size=pool_size, 
#            used_correct_pool_inputs=used_correct_pool_inputs)


#def _get_n_pixels_affected_by_padding(network_depth, conv_size=(3,3,3), 
#                                     pool_size=(2,2,2)):
#    """
#    Determine the effective number of pixels that are affected by padding
#    in unet.
#    
#    Args:
#        network_depth (int): network_depth param of unet
#        conv_size (3-tuple): size of a convolution kernel.  Conv is assumed
#            to act on the spatial dimensions only.
#        pool_size (3-tuple): size of pooling and upsampling (see notes).  
#            Pooling and Upsampling are assumed to act on the spatial 
#            dimensions only.
#    
#    Returns:
#        pixels_lost (array-like, 5d): number of pixels affected by padding
#        in each dimension due to the padding at the border of convolutions.  
#        Half of the pixels would be on the front/upper/left borders, the other 
#        half on the back/lower/right borders.
#    
#    Notes:
#        upsam_size is assumed to be the same as pool_size.  Otherwise
#        pixels do not correspond during concatenation in skip-connections.
#        
#        Upsampling is assumed to be either 
#        - a transposed conv with strides "upsam_size" or 
#        - FT or NN upsampling followed by a conv.  
#        Both lead to the same output shape.
#        
#        The effective number of affected pixels is high, because pixels
#        in lower layers are later upsampled.  Thus, if one pixel is affected in
#        bottom layer, 2**(network_depth-1) pixels are affected in final image.
#    """
#    conv_size = convert_shape_to_np_array(conv_size)
#    pool_size = convert_shape_to_np_array(pool_size)
#    upsam_size = pool_size  # otherwise pixels don't correspond during concat.
#    
#    first_affected = 2*(conv_size-1)  # 2 convs in first block
#    def down_affected(layer_index):
#        # 2 convs, 1 pix in layer == pool_size**layer_index upsampled pix's
#        return 2*(conv_size-1) * pool_size**layer_index
#    # 2 convs, 1 pix == pool_size**net_depth upsampled pix's
#    bottom_affected = 2*(conv_size-1) * pool_size**(network_depth-1)  
#    def up_affected(layer_index):
#        # 3 convs, 1 pix == pool_isze**layer_index upsampled pix's
#        return 3*(conv_size-1) * upsam_size**layer_index  
#    final_affected = 4*(conv_size-1)  # 4 convs
#    
#    pixels_affected_in_spatial_dim = 0 
#    pixels_affected_in_spatial_dim += first_affected
#    for layer_index in range(1, network_depth-1):
#        pixels_affected_in_spatial_dim += down_affected(layer_index)
#    pixels_affected_in_spatial_dim += bottom_affected
#    for layer_index in range(1, network_depth-1):
#        pixels_affected_in_spatial_dim += up_affected(layer_index)
#    pixels_affected_in_spatial_dim += final_affected
#    
#    res = np.zeros(5, dtype=np.int)
#    for i in range(3):
#        res[i+1] = pixels_affected_in_spatial_dim[i]
#    return res


# %% get output shape of entire unet

#def get_net_output_shape(in_shape, network_depth, padding, conv_size=(3,3,3), 
#                         pool_size=(2,2,2)):
#    """get output shape after fwd-prop through unet"""
#    # 64 for n_channels_start is arbitrary.  Last layer will always have 1.
#    # for same padding, this should be the same as the in_shape
#    return get_block_output_shapes(in_shape, 64, network_depth, padding,
#                                   conv_size, pool_size)[-1]


# %% get output shapes of every unet-block as list

# STOPPED USING THESE IN FAVOR OF GOING LAYER BY LAYER
# TODO: ONLY IT MIGHT MAKE SENSE TO USE THESE FOR CONV-BLOCKS
# OR RESNET-BLOCKS ONCE THEY ARE IMPLEMENTED.

#def get_block_output_shapes(in_shape, n_channels_start, network_depth, padding, 
#                            conv_size=(3,3,3), pool_size=(2,2,2)):
#    """get shape after each block after fwd-prop through unet as list."""
#    if not is_allowed_input_shape(in_shape, network_depth, padding, 
#                                  conv_size, pool_size):
#        raise ValueError("in_shape " + str(in_shape) + " is not allowed " + 
#                         "for unet with the given depth and padding.")
#    if network_depth < 2:
#        raise ValueError("network depth must be greater than 1 for unet.")
#    # layer 0
#    sh = get_first_block_output_shape(in_shape, n_channels_start, padding)
#    layer_out_shapes = [list(sh),]
#    # layers 1...depth-2
#    for layer_index in range(1, network_depth-1):
#        sh = get_down_block_output_shape(sh, padding)
#        layer_out_shapes.append(list(sh))
#        
#    # layer depth-1 (last of down-conv // first of up-conv)
#    sh = get_bottom_block_output_shape(sh, padding)
#    layer_out_shapes.append(list(sh))
#    
#    # indexing backwards for elegant implementation
#    # layers depth-2 ... 1
#    for layer_index in range(network_depth-2, 0, -1):
#        sh = get_up_block_output_shape(sh, padding)
#        layer_out_shapes.append(list(sh))
#    # layer 0 (final)
#    sh = get_final_block_output_shape(sh, padding)
#    layer_out_shapes.append(list(sh))
#
#    return layer_out_shapes
#
#
# %% get output shape after blocks in unet

#def get_block_sequence_output_shape(in_shape, blocks, n_channels_start, padding,
#                    conv_size=(3,3,3), pool_size=(2,2,2), upsam_size=(2,2,2)):
#    """
#    Determine the output shape after a sequence of blocks in unet.
#    
#    Args:
#        in_shape (array-like, 5d): shape of a batch in the form 
#            (n_images, depth, height, width, channels)
#        n_channels_start (int): number of output channels of first conv
#        padding (str): "same" or "valid"
#        blocks: tuple of strings such as ("first_block", "down_block", 
#            "bottom_block", "up_block", "final_block")
#        conv_size (3-tuple): size of a convolution kernel.  Conv is assumed
#            to act on the spatial dimensions only.
#        upsam_size (3-tuple): size of upsampling (see notes).  Upsampling is 
#            assumed to act on the spatial dimensions only.
#    
#    Returns:
#        out_shape (array-like, 5d): shape after the sequence of blocks.
#    
#    Notes:
#        Upsampling is assumed to be either 
#        - a transposed conv with strides "upsam_size" or 
#        - FT or NN upsampling followed by a conv.  
#        Both lead to the same output shape.
#    """
#    shape = in_shape
#    for block in blocks:
#        if block == "first_block":
#            shape = get_first_block_output_shape(shape, n_channels_start, 
#                        padding, conv_size=conv_size, pool_size=pool_size)
#        elif block == "down_block":
#            shape = get_down_block_output_shape(shape, padding, 
#                        conv_size=conv_size, pool_size=pool_size) 
#        elif block == "bottom_block":
#            shape = get_bottom_block_output_shape(shape, padding, 
#                        conv_size=conv_size) 
#        elif block == "up_block":
#            shape = get_up_block_output_shape(shape, padding, 
#                        conv_size=conv_size, upsam_size=upsam_size)  
#        elif block == "final_block":
#            shape = get_final_block_output_shape(shape, padding, 
#                        conv_size=conv_size, upsam_size=upsam_size)
#    return shape
#
#def get_first_block_output_shape(in_shape, n_channels_start, padding, 
#                                 conv_size=(3,3,3), pool_size=(2,2,2)):
#    """
#    Determine the output shape after the first unet-block.
#    
#    Args:
#        in_shape (array-like, 5d): shape of a batch in the form 
#            (n_images, depth, height, width, channels)
#        n_channels_start (int): number of output channels of first conv
#        padding (str): "same" or "valid"
#        conv_size (3-tuple): size of a convolution kernel.  Conv is assumed
#            to act on the spatial dimensions only.
#        pool_size (3-tuple): size of pooling.  Pooling is assumed
#            to act on the spatial dimensions only.
#    
#    Returns:
#        out_shape (array-like, 5d): shape after the block.
#    """
#    sequence = ("conv", "conv", "pool")
#    out_num = n_channels_start
#    return get_layer_sequence_output_shape(in_shape, sequence, out_num, 
#                            padding, conv_size=conv_size, pool_size=pool_size)
#
#def get_down_block_output_shape(in_shape, padding, conv_size=(3,3,3), 
#                                pool_size=(2,2,2)):
#    """
#    Determine the output shape after the down-blocks in unet.
#    
#    Args:
#        in_shape (array-like, 5d): shape of a batch in the form 
#            (n_images, depth, height, width, channels)
#        padding (str): "same" or "valid"
#        conv_size (3-tuple): size of a convolution kernel.  Conv is assumed
#            to act on the spatial dimensions only.
#        pool_size (3-tuple): size of pooling.  Pooling is assumed
#            to act on the spatial dimensions only.
#    
#    Returns:
#        out_shape (array-like, 5d): shape after the block.
#    """
#    sequence = ("conv", "conv", "pool")
#    out_num = 2*in_shape[-1]
#    return get_layer_sequence_output_shape(in_shape, sequence, out_num, padding,
#                                     conv_size=conv_size, pool_size=pool_size)
#
#def get_bottom_block_output_shape(in_shape, padding, conv_size=(3,3,3)):
#    """
#    Determine the output shape after the bottom-block in unet.
#    
#    Args:
#        in_shape (array-like, 5d): shape of a batch in the form 
#            (n_images, depth, height, width, channels)
#        padding (str): "same" or "valid"
#        conv_size (3-tuple): size of a convolution kernel.  Conv is assumed
#            to act on the spatial dimensions only.
#    
#    Returns:
#        out_shape (array-like, 5d): shape after the block.
#    """
#    sequence = ("conv", "conv")
#    out_num = 2*in_shape[-1]
#    return get_layer_sequence_output_shape(in_shape, sequence, out_num, padding,
#                                     conv_size=conv_size)
#
#def get_up_block_output_shape(in_shape, padding, conv_size=(3,3,3),
#                              upsam_size=(2,2,2)):
#    """
#    Determine the output shape after the up-blocks in unet.
#    
#    Args:
#        in_shape (array-like, 5d): shape of a batch in the form 
#            (n_images, depth, height, width, channels)
#        padding (str): "same" or "valid"
#        conv_size (3-tuple): size of a convolution kernel.  Conv is assumed
#            to act on the spatial dimensions only.
#        upsam_size (3-tuple): size of upsampling (see notes).  Upsampling is 
#            assumed to act on the spatial dimensions only.
#    
#    Returns:
#        out_shape (array-like, 5d): shape after the block.
#    
#    Notes:
#        Upsampling is assumed to be either 
#        - a transposed conv with strides "upsam_size" or 
#        - FT or NN upsampling followed by a conv.  
#        Both lead to the same output shape.
#    """
#    sequence = ("upsam", "conv", "conv")
#    if in_shape[-1] % 2 != 0:
#        raise ValueError("input to upsampling layer must have even number " +
#                         "of channels.")
#    out_num = in_shape[-1] // 2
#    return get_layer_sequence_output_shape(in_shape, sequence, out_num, padding,
#                                   conv_size=conv_size, upsam_size=upsam_size)
#
#def get_final_block_output_shape(in_shape, padding, conv_size=(3,3,3), 
#                                 upsam_size=(2,2,2)):
#    """
#    Determine the output shape after the final-block in unet.
#    
#    Args:
#        in_shape (array-like, 5d): shape of a batch in the form 
#            (n_images, depth, height, width, channels)
#        padding (str): "same" or "valid"
#        conv_size (3-tuple): size of a convolution kernel.  Conv is assumed
#            to act on the spatial dimensions only.
#        upsam_size (3-tuple): size of upsampling (see notes).  Upsampling is 
#            assumed to act on the spatial dimensions only.
#    
#    Returns:
#        out_shape (array-like, 5d): shape after the block.
#    
#    Notes:
#        Upsampling is assumed to be either 
#        - a transposed conv with strides "upsam_size" or 
#        - FT or NN upsampling followed by a conv.  
#        Both lead to the same output shape.
#    """
#    sequence = ("upsam", "conv", "conv", "conv")
#    out_num = 1
#    return get_layer_sequence_output_shape(in_shape, sequence, out_num, padding,
#                                   conv_size=conv_size, upsam_size=upsam_size)



# %% get output shape after layers of unet
def get_layer_sequence_output_shape(in_shape, sequence, out_num, padding,
                    conv_size=(3,3,3), pool_size=(2,2,2), upsam_size=(2,2,2),
                    im_axis=0, channel_axis=-1):
    """
    Determine the output shape after a sequence of layers.
    
    Args:
        in_shape (array-like, 5d): shape of a batch in the form 
            (n_images, depth, height, width, channels)
        sequence: a tuple of strings such as ("conv", "upsam", "pool")
        out_num (int): number of output channels of last layer in sequence.
            NOTE: this will be ignored if only pooling ops are used!
        padding (str): "valid" or "same"
        conv_size (3-tuple): size of a convolution kernel.  Conv is assumed
            to act on the spatial dimensions only.
        pool_size (3-tuple): size of pooling.  Pooling is assumed
            to act on the spatial dimensions only.
        upsam_size (3-tuple): size of upsampling (see notes).  Upsampling is 
            assumed to act on the spatial dimensions only.
    
    Returns:
        out_shape (array-like, 5d): shape after the sequence of layers.
    
    Notes:
        - Upsampling is assumed to be either FT or NN upsampling.  
        - For upsampling via strided conv, use this followed by 
          get_conv_layer_output_shape
        - should in principle also work for 2d-images, but this has not been
          tested.
        - For the default values in 3d, the input and output shapes should be 
          (n_images, depth, height, width, channels)
        
        For unet, pool_size should normally be the same as upsam_size
    """
    # TODO: change arg order and add checks for empty out_num for the case
    # that only pooling ops are used.
    out_shape = in_shape  # init
    for layer in sequence:
        if layer == "conv":
            out_shape = get_conv_output_shape(out_shape, out_num, padding,
              conv_size=conv_size, im_axis=im_axis, channel_axis=channel_axis)
        elif layer == "pool":
            out_shape = get_pool_output_shape(out_shape, padding, 
              pool_size=pool_size, im_axis=im_axis, channel_axis=channel_axis)
        elif layer == "upsam":
            out_shape = get_upsam_output_shape(out_shape, upsam_size=upsam_size,
              im_axis=im_axis, channel_axis=channel_axis)
        elif layer == "concat":
            in_shape = get_concat_output_shape(out_shape, 
              channel_axis=channel_axis)
        else:
            raise ValueError("unkown layer '" + layer + "'.")
    return out_shape


def get_conv_output_shape(in_shape, out_num, padding, conv_size=(3,3,3),
                          im_axis=0, channel_axis=-1):
    """
    Determine the output shape of a convolution layer for a given input shape.
    
    Args:
        in_shape (array-like, 5d): shape of a batch in the form 
            (n_images, depth, height, width, channels)
        out_num (int): number of output_channels
        padding (str): "same" or "valid"
        conv_size (3-tuple): size of a convolution kernel.  Conv is assumed
            to act on the spatial dimensions only.
    
    Returns:
        out_shape (array-like, 5d): shape after convolution layer.

    Notes:
        - should in principle also work for 2d-images, but this has not been
          tested.
        - For the default values in 3d, the input and output shapes should be 
          (n_images, depth, height, width, channels)
    """
    if padding == "same":
        shape = convert_shape_to_np_array(in_shape)
        shape[channel_axis] = out_num
    elif padding == "valid":
        shape = _get_valid_conv_output_shape(in_shape, out_num, 
               conv_size=conv_size, im_axis=im_axis, channel_axis=channel_axis)
    return shape

def _get_valid_conv_output_shape(in_shape, out_num, conv_size=(3,3,3),
                                 im_axis=0, channel_axis=-1):
    """
    Determine the output shape of a VALID convolution for a given input shape.
    output shape is smaller because (kernel_size-1) pixels are lost at the 
    border.
    
    Args:
        in_shape (array-like, 5d): shape of a batch in the form 
            (n_images, depth, height, width, channels)
        out_num (int): number of output_channels
        conv_size (3-tuple): size of a convolution kernel.  Conv is assumed
            to act on the spatial dimensions and feature dimension.
        im_axis (int): axis along which image batches were formed (usually 0)
        channel_axis (int): axis along which feature channels are stored.
            If 'channels_first', this is 1 and for 'channels_last' it is -1.

    Returns:
        out_shape (array-like, 5d): shape after valid convolution.
        
    Notes:
        - should in principle also work for 2d-images, but this has not been
          tested.
        - For the default values in 3d, the input and output shapes should be 
          (n_images, depth, height, width, channels)
    """
    border_size = convert_shape_to_np_array(conv_size) - 1
    spatial_axes = get_spatial_axes(mode="batch", ndims=len(in_shape)-2,
                                    im_axis=im_axis, channel_axis=channel_axis)
    shape = convert_shape_to_np_array(in_shape)
    shape[spatial_axes] = shape[spatial_axes] - border_size
    shape[channel_axis] = out_num
    return shape

def get_pool_output_shape(in_shape, padding, pool_size=(2,2,2), im_axis=0,
                          channel_axis=-1):
    """
    Determine the output shape of a pooling layer for a given input shape.
    
    Args:
        in_shape (array-like, 5d): shape of a batch in the form 
            (n_images, depth, height, width, channels)
        padding (str): "same" or "valid"
        pool_size (3-tuple): size of pooling.  Pooling is assumed
            to act on the spatial dimensions only.
        im_axis (int): axis along which image batches were formed (usually 0)
        channel_axis (int): axis along which feature channels are stored.
            If 'channels_first', this is 1 and for 'channels_last' it is -1.
    
    Returns:
        out_shape (array-like, 5d): shape after pooling layer.
       
    Notes:
        - should in principle also work for 2d-images, but this has not been
          tested.
        - For the default values in 3d, the input and output shapes should be 
          (n_images, depth, height, width, channels)
    """
    shape = convert_shape_to_np_array(in_shape)
    pool_size = convert_shape_to_np_array(pool_size)
    spatial_axes = get_spatial_axes(mode="batch", ndims=len(in_shape)-2,
                                    im_axis=im_axis, channel_axis=channel_axis)
    spatial_shape = shape[spatial_axes].astype(np.int)
    if padding == "same":
        # TODO: is this safe for floating point arithmetic?
        shape[spatial_axes] = np.ceil(spatial_shape / pool_size).astype(np.int)
    elif padding == "valid":
        shape[spatial_axes] = shape[spatial_axes] // pool_size # floor
    return shape

def get_upsam_output_shape(in_shape, upsam_size=(2,2,2), im_axis=0, 
                           channel_axis=-1):
    """
    Determine the output shape of an upsampling layer for a given input shape.  
    
    Args:
        in_shape (array-like, 5d): shape of a input batch
        out_num (int): number of output channels
        upsam_size (3-tuple): size of upsampling (see notes).  Upsampling is 
            assumed to act on the spatial dimensions only.
        im_axis (int): axis along which image batches were formed (usually 0)
        channel_axis (int): axis along which feature channels are stored.
            If 'channels_first', this is 1 and for 'channels_last' it is -1.
    
    Returns:
        out_shape (array-like, 5d): shape after upsampling-layer.
    
    Notes:
        - Upsampling is assumed to be either FT or NN upsampling.  
        - For upsampling via strided conv, use this followed by 
          get_conv_layer_output_shape
        - should in principle also work for 2d-images, but this has not been
          tested.
        - For the default values in 3d, the input and output shapes should be 
          (n_images, depth, height, width, channels)
    """
    shape = convert_shape_to_np_array(in_shape)
    upsam_size = convert_shape_to_np_array(upsam_size)
    spatial_axes = get_spatial_axes(mode="batch", ndims=len(in_shape)-2,
                                    im_axis=im_axis, channel_axis=channel_axis)
    shape[spatial_axes] = shape[spatial_axes] * upsam_size
    return shape

def get_concat_output_shape(in_shape, channel_axis=-1):
    """
    Determine shape after concatenation along channels-dimension assuming
    that both input arrays have same shape (spatial and n_channels).
    
    Args:
        in_shape (array-like, 5d): shape of an input batch
        channel_axis (int): axis along which feature channels are stored.
            If 'channels_first', this is 1 and for 'channels_last' it is -1.
        
    Returns:
        out_shape (array-like, 5d): shape after concatenation along 
        channel_axis.
    """
    return _get_concat_output_shape(in_shape, in_shape, 
                                    channel_axis=channel_axis)

def _get_concat_output_shape(in_shape1, in_shape2, channel_axis=-1):
    """
    Determine shape after concatenation along channel-axis.  Concatenation
    of both input arrays must be possible, i.e. in_shape1 and in_shape2
    must be the same except for channel_axis
    
    Args:
        in_shape1 (array-like, 5d): shape of an input batch
        in_shape2 (array-like, 5d): shape of an input batch
        channel_axis (int): axis along which feature channels are stored.
            If 'channels_first', this is 1 and for 'channels_last' it is -1.
        
    Returns:
        out_shape (array-like, 5d): shape after concatenation along 
        channel_axis.
        
    Raises ValueError, if the input shapes are not equal in all dimensions 
    except the channels dimension.  This might raise an error, if one of the 
    shapes is None, but where concat with tf is possible.
    """
    in_shape1 = convert_shape_to_np_array(in_shape1)
    in_shape2 = convert_shape_to_np_array(in_shape2)
    not_channels = np.arange(len(in_shape1)) != channel_axis
    if np.any(in_shape1[not_channels] != in_shape2[not_channels]):
       print(in_shape1, in_shape2)
       raise ValueError("Shape along the other dimensions is not equal. " +
                        "I do not know how to continue. ")
    out_shape = in_shape1
    out_shape[channel_axis] = in_shape1[channel_axis] + in_shape2[channel_axis]
    return out_shape


# %% inverse ops calculate shape before layer or block

# STOPPED USING THESE IN FAVOR OF GOING LAYER BY LAYER
# TODO: ONLY IT MIGHT MAKE SENSE TO USE THESE FOR CONV-BLOCKS
# OR RESNET-BLOCKS ONCE THEY ARE IMPLEMENTED.
    
## %% get input shape before blocks in unet
#def get_block_sequence_input_shape(out_shape, blocks, n_channels_start, padding, 
#                    conv_size=(3,3,3), pool_size=(2,2,2), upsam_size=(2,2,2),
#                    used_correct_pool_inputs=False):
#    """
#    Inverse operation to get_shape_valid_blocks.
#    Calculates the number of pixels the was there before the sequence of ops
#    See that function for info about params.
#    
#    NOTE: The sequence must be as in the fwd-pass.  It will be reversed!
#    
#    CAUTION: 
#    In general, the input shape of a pooling layer cannot be infered from its
#    output shape, since several inputs map to the same output (the exact range
#    depends on the padding). This function will issue a warning and simply 
#    multiply the number of pixels in each spatial dimension by pool sizes.  
#    
#    You can disable the warning by declaring that you used an input that 
#    left no border while pooling (i.e. in_shape[i] % pool_size[i] == 0 
#    for all spatial dimensions.).
#    """
#    
#    # TODO: make n_channels start kwarg.  It is not always necessary.
#    # assert that output channels of first layer is n_channels_start
#    shape = out_shape
#    blocks = blocks[::-1]  # run through blocks in reverse
#    for block in blocks:
#        if block == "final_block":
#            shape = get_final_block_input_shape(shape, n_channels_start, 
#                        padding, conv_size=conv_size, upsam_size=upsam_size)
#        elif block == "up_block":
#            shape = get_up_block_input_shape(shape, padding, 
#                        conv_size=conv_size, upsam_size=upsam_size)
#        elif block == "bottom_block":
#            shape = get_bottom_block_input_shape(shape, padding, 
#                        conv_size=conv_size)
#        elif block == "down_block":
#            shape = get_down_block_input_shape(shape, padding, 
#                        conv_size=conv_size, pool_size=pool_size,
#                        used_correct_pool_inputs=used_correct_pool_inputs)  
#        elif block == "first_block":
#            shape = get_first_block_input_shape(shape, padding,
#                        conv_size=conv_size, pool_size=pool_size,
#                        used_correct_pool_inputs=used_correct_pool_inputs)
#    return shape
#
#def get_first_block_input_shape(out_shape, padding, conv_size=(3,3,3), 
#                        pool_size=(2,2,2), used_correct_pool_inputs=False):
#    """
#    Inverse operation to get_first_block_output_shape.
#    Calculates the shape before the block.
#    See that function for info about params.
#    
#    CAUTION: 
#    In general, the input shape of a pooling layer cannot be infered from its
#    output shape, since several inputs map to the same output (the exact range
#    depends on the padding). This function will issue a warning and simply 
#    multiply the number of pixels in each spatial dimension by pool sizes.  
#    
#    You can disable the warning by declaring that you used an input that 
#    left no border while pooling (i.e. in_shape[i] % pool_size[i] == 0 
#    for all spatial dimensions.).
#    """
#    sequence = ("conv", "conv", "pool")
#    in_num = 1  # assumes black and white input.
#    return get_layer_sequence_input_shape(out_shape, sequence, in_num, padding, 
#                            conv_size=conv_size, pool_size=pool_size,
#                            used_correct_pool_inputs=used_correct_pool_inputs)
#
#def get_down_block_input_shape(out_shape, padding, conv_size=(3,3,3), 
#                           pool_size=(2,2,2), used_correct_pool_inputs=False):
#    """
#    Inverse operation to get_down_block_output_shape.
#    Calculates the shape before the block.
#    See that function for info about params.
#    
#    CAUTION: 
#    In general, the input shape of a pooling layer cannot be infered from its
#    output shape, since several inputs map to the same output (the exact range
#    depends on the padding). This function will issue a warning and simply 
#    multiply the number of pixels in each spatial dimension by pool sizes.  
#    
#    You can disable the warning by declaring that you used an input that 
#    left no border while pooling (i.e. in_shape[i] % pool_size[i] == 0 
#    for all spatial dimensions.).
#    """
#    sequence = ("conv", "conv", "pool")
#    in_num = 0.5*out_shape[-1]
#    return get_layer_sequence_input_shape(out_shape, sequence, in_num, padding, 
#                            conv_size=conv_size, pool_size=pool_size,
#                            used_correct_pool_inputs=used_correct_pool_inputs)
#
#def get_bottom_block_input_shape(out_shape, padding, conv_size=(3,3,3)):
#    """
#    Inverse operation to get_bottom_block_output_shape.
#    Calculates the shape before the block.
#    See that function for info about params.
#    """
#    sequence = ("conv", "conv")
#    in_num = 0.5*out_shape[-1]
#    return get_layer_sequence_input_shape(out_shape, sequence, in_num, padding,
#                                          conv_size=conv_size)
#
#def get_up_block_input_shape(out_shape, padding, conv_size=(3,3,3),
#                             upsam_size=(2,2,2)):
#    """
#    Inverse operation to get_up_block_output_shape.
#    Calculates the shape before the block.
#    See that function for info about params.
#    """
#    sequence = ("upsam", "conv", "conv")
#    in_num = 2*out_shape[-1]
#    return get_layer_sequence_input_shape(out_shape, sequence, in_num, padding,
#                                   conv_size=conv_size, upsam_size=upsam_size)
#
#def get_final_block_input_shape(in_shape, in_num, padding, conv_size=(3,3,3), 
#                                 upsam_size=(2,2,2)):
#    """
#    Inverse operation to get_final_block_output_shape.
#    Calculates the shape before the block.
#    See that function for info about params.
#    """
#    sequence = ("upsam", "conv", "conv", "conv")
#    return get_layer_sequence_output_shape(in_shape, sequence, in_num, padding,
#                                   conv_size=conv_size, upsam_size=upsam_size)


# %% get input shape before layers in unet

# TODO: implement these in terms of get_output_shape with negative numbers
# or implement a new abstraction layer to be used by both?

def get_layer_sequence_input_shape(out_shape, sequence, in_num, padding, 
            conv_size=(3,3,3), pool_size=(2,2,2), upsam_size=(2,2,2), 
            used_correct_pool_inputs=False, im_axis=0, channel_axis=-1):
    """    
    Inverse operation to get_layer_sequence_output_shape.
    Calculates the number of pixels the was there before the sequence of layers.
    See get_layer_sequence_output_shape for info about params.
    
    NOTE: The sequence must be as in the fwd-pass.  It will be reversed!
    
    CAUTION: 
    In general, the input shape of a pooling layer cannot be infered from its
    output shape, since several inputs map to the same output (the exact range
    depends on the padding). This function will issue a warning and simply 
    multiply the number of pixels in each spatial dimension by pool sizes.  
    You can disable the warning by declaring that you used an input that 
    left no border while pooling (i.e. in_shape[i] % pool_size[i] == 0 
    for all layers and spatial dimensions.).
    """
    sequence = sequence[::-1]
    in_shape = out_shape  # init
    for layer in sequence:
        if layer == "conv":
            # careful: in_num will only be correct for last layer in seq
            in_shape = get_conv_input_shape(in_shape, in_num, padding,
              conv_size=conv_size, im_axis=im_axis, channel_axis=channel_axis)
        elif layer == "pool":
            in_shape = get_pool_input_shape(in_shape, pool_size=pool_size, 
              used_correct_input=used_correct_pool_inputs, 
              im_axis=im_axis, channel_axis=channel_axis)
        elif layer == "upsam":
            in_shape = get_upsam_input_shape(in_shape, upsam_size=upsam_size, 
              im_axis=im_axis, channel_axis=channel_axis)
        elif layer == "concat":
            in_shape = get_concat_input_shape(out_shape, 
              channel_axis=channel_axis)
        else:
            raise ValueError("layer '" + layer + "' is unkown.")
    return in_shape

def get_conv_input_shape(out_shape, in_num, padding, conv_size=(3,3,3),
                         im_axis=0, channel_axis=-1):
    """
    Inverse operation to get_conv_output_shape.
    Calculates the number of pixels the was there before the conv-operation.  
    See get_pool_output_shape for info about params.
    """
    if padding == "same":
        shape = out_shape
        shape[channel_axis] = in_num
    elif padding == "valid":
        shape = _get_valid_conv_input_shape(out_shape, in_num, 
              conv_size=conv_size, im_axis=im_axis, channel_axis=channel_axis)
    return shape

def _get_valid_conv_input_shape(out_shape, in_num, conv_size=(3,3,3), 
                                im_axis=0, channel_axis=-1):
    """
    Inverse operation to _get_valid_conv_output_shape.
    Calculates the number of pixels the was there before the conv-operation
    with valid padding.  See get_shape_valid_conv for info about params.
    """
    border_size = convert_shape_to_np_array(conv_size) - 1
    shape = convert_shape_to_np_array(out_shape)
    spatial_axes = get_spatial_axes(mode="batch", ndims=len(out_shape)-2,
                                    im_axis=im_axis, channel_axis=channel_axis)
    shape[spatial_axes] = shape[spatial_axes] + border_size
    shape[channel_axis] = in_num
    return shape

def get_pool_input_shape(out_shape, pool_size=(2,2,2), used_correct_input=False,
                         im_axis=0, channel_axis=-1):
    """
    CAUTION: 
    In general, the input shape of a pooling layer cannot be infered from its
    output shape, since several inputs map to the same output (the exact range
    depends on the padding). This function will issue a warning and simply 
    multiply the number of pixels in each spatial dimension by pool sizes.  
    
    You can disable the warning by declaring that you used an input that 
    left no border while pooling (i.e. in_shape[i] % pool_size[i] == 0 
    for all spatial dimensions.).
    
    Inverse operation to get_pool_output_shape.
    Calculates the number of pixels that was there before the pool-operation.  
    See get_pool_output_shape for info about params.
    """
    # TODO: maybe this should raise an exception
    if not used_correct_input:
        warn("Input shape of a pooling is ambiguous.  See docs of " +
             "get_pool_input_shape on when and how to disable this warning.")
    shape = convert_shape_to_np_array(out_shape)
    pool_size = convert_shape_to_np_array(pool_size)
    spatial_axes = get_spatial_axes(mode="batch", ndims=len(out_shape)-2,
                                    im_axis=im_axis, channel_axis=channel_axis)
    shape[spatial_axes] = shape[spatial_axes] * pool_size
    return shape

def get_upsam_input_shape(out_shape, upsam_size=(2,2,2), 
                          im_axis=0, channel_axis=-1):
    """
    Inverse operation to get_upsam_output_shape.
    Calculates the number of pixels the was there before the upsam-operation.
    See get_upsam_output_shape for info about params.
    """
    shape = convert_shape_to_np_array(out_shape)
    upsam_size = convert_shape_to_np_array(upsam_size)
    spatial_axes = get_spatial_axes(mode="batch", ndims=len(out_shape)-2,
                                    im_axis=im_axis, channel_axis=channel_axis)
    shape[spatial_axes] = shape[spatial_axes] // upsam_size  # TODO check for remainder?
    return shape

def get_concat_input_shape(out_shape, channel_axis=-1):
    """
    gives input shape before concatenation along channels-dimension assuming
    that both arrays had same shape (spatial and n_channels)
    before concatenation.
    out_shape (array_like) is the shape that was the output of concatenation
    channel_axis depends on "channels_last" / "channels_first"
    """
    out_shape = convert_shape_to_np_array(out_shape)
    return _get_concat_input_shapes(out_shape, out_shape[channel_axis]//2, 
                                    channel_axis=channel_axis)

def _get_concat_input_shapes(out_shape, in_num1, channel_axis=-1):
    """
    gives input shapes (as tuple) before concatenation along channels-dimension.
    out_shape (array_like) is the shape that was the output of concatenation 
    in_num1 (int) is the number of channels of the first array.
    """
    if in_num1 > out_shape[channel_axis]:
        raise ValueError("in_num too large. The first input array can't have "+
                         "more channels than the concatenated array.")
    in_shape1 = out_shape
    in_shape1[channel_axis] = in_num1
    in_shape2 = out_shape
    in_shape2[channel_axis] = out_shape[channel_axis] - in_num1
    return in_shape1, in_shape2


# %% misc
def convert_shape_to_np_array(sh):
    """converts list, tuple, ... and also tf.TensorShape to np-array"""
    if isinstance(sh, tf.TensorShape):
        sh = sh.as_list()
    return np.array(sh)

def get_spatial_axes(mode="single", ndims=3, im_axis=0, channel_axis=-1):
    """
    mode: 'single' image or entire 'batch'
    """
    # TODO consider removing single mode as special case of batch mode
    if mode not in ["single", "batch"]:
        raise ValueError("Unknown mode '" + mode + "'." )
    if im_axis < 0:  # this should never occur as im_axis should be 0
        im_axis = (ndims + 2) + im_axis
    if channel_axis < 0:
        channel_axis = (ndims + 2) + channel_axis
    
    spatial_axes = list()
    if mode == "batch":
        for i in range(ndims + 2):
            if i not in [im_axis, channel_axis]:
                spatial_axes.append(i)
    elif mode == "single":
        if im_axis < channel_axis:
            for i in range(ndims + 1):
                if i != channel_axis-1:
                    spatial_axes.append(i)
        elif im_axis > channel_axis:  # this should never occur as im_axis should be 0
            for i in range(ndims + 1):
                if i != channel_axis:
                    spatial_axes.append(i)

    return spatial_axes

