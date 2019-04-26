#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:11:24 2018

@author: soenke
"""
import tensorflow as tf

#for relu as in https://arxiv.org/pdf/1502.01852.pdf
he_init = tf.variance_scaling_initializer(scale=2.0, mode="fan_in", 
                                          distribution="normal")

trunc_normal_init = tf.truncated_normal_initializer()