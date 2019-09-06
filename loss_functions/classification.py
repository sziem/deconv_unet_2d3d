# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 14:25:21 2018

@author: photon
"""

#import tensorflow as tf
# for tensorflow 1.14 use this to avoid warnings:
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# TODO: is there any advantage in using sparse_softmax_cross_entropy with labellist?
def softmax_cross_entropy(
        y_batch_onehot, y_predicted_batch, weights=1.0, label_smoothing=0):
    """ Docstring TODO """
    # y_batch should be one_hot tensor
    # y_predicted_batch should be unactivated.  softmax is applied to it here.
    return tf.losses.softmax_cross_entropy(
            y_batch_onehot, y_predicted_batch, weights=weights, 
            label_smoothing=label_smoothing)
    
def hinge_loss(
        y_batch_onehot, y_predicted_batch, weights=1.0):
    """ Docstring TODO """
    # y_batch should be one_hot tensor with values 1.0 and 0.0
    # y_predicted_batch should be unactivated.  softmax is applied to it here.
    return tf.losses.hinge_loss(
            y_batch_onehot, y_predicted_batch, weights=weights)
