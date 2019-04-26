#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 17:19:40 2018

@author: soenke
"""        

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MaxAbsScaler, check_array

def maxabs_scale(X, axis=0, copy=True):
    """Scale each feature to the [0, 1] range without breaking the sparsity.

    This estimator scales each feature individually such
    that the maximal absolute value of each feature in the
    training set will be 1.0.

    This scaler can also be applied to sparse CSR or CSC matrices.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data.

    axis : int (0 by default)
        axis used to scale along. If 0, independently scale each feature,
        otherwise (if 1) scale each sample.

    copy : boolean, optional, default is True
        Set to False to perform inplace scaling and avoid a copy (if the input
        is already a numpy array).

    See also
    --------
    MaxAbsScaler: Performs scaling to the [-1, 1] range using the``Transformer`` API
        (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).

    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """  # noqa
    # Unlike the scaler object, this function allows 1d input.

    # If copy is required, it will be done inside the scaler object.
    X = check_array(X, accept_sparse=('csr', 'csc'), copy=False,
                    ensure_2d=False, dtype=FLOAT_DTYPES)
    original_ndim = X.ndim

    if original_ndim == 1:
        X = X.reshape(X.shape[0], 1)

    s = MaxAbsScaler(copy=copy)
    if axis == 0:
        X = s.fit_transform(X)
    else:
        X = s.fit_transform(X.T).T

    if original_ndim == 1:
        X = X.ravel()

    return X







#np_x = np.arange(36).reshape((4,3,3))
np_x = np.arange(12).reshape((4,3))
# np_x_norm = scale(np_x, axis=1, with_mean=False, with_std=False, copy=True)
np_x_norm = minmax_scale(np_x, axis=0)
# np_x_norm = maxabs_scale(np_x, axis=0)
print(np_x)
print(np_x_norm)









##X = scale( X, axis=0, with_mean=True, with_std=True, copy=True )
#
#
##def np_normalize_along_axis_2d_neg1_1(t, axis):
##    t = np.expand_dims(t,np.newaxis)
##    return 2 * (t-np.min(t, axis=axis)) / np.max(t, axis=axis) - 1
#
##def l2_reg(im):
##    # ready for 3D!
##    """
##    aka intensity penalty
##    aka carrington
##    
##    Args: 
##        im (tf-tensor, 2d): image
##        
##    Returns:
##        l2_regularization (float, 1d):  sum of square of pixel values
##    """
##    # Can also be used in case some values become negative.
##    # then increase values and add intensity penalty  
##    # in other case:  spread out values?
##    return tf.reduce_sum(im**2)
#
##def np_normalize_0_1_global(t):
##    return 2 * (t-t.min()) / t.max() - 1
#
#    X = check_array(X, accept_sparse=('csr', 'csc'), copy=False,
#                    ensure_2d=False, dtype=FLOAT_DTYPES)
#    original_ndim = X.ndim
#
#    if original_ndim == 1:
#        X = X.reshape(X.shape[0], 1)
#
#    s = MaxAbsScaler(copy=copy)
#    if axis == 0:
#        X = s.fit_transform(X)
#    else:
#        X = s.fit_transform(X.T).T
#
#    if original_ndim == 1:
#        X = X.ravel()
#
#    return X
#
#
#def np_normalize_0_1(t, axis=None):
#    return t / np.max(np.abs(t), axis=axis)
#
#
#
## %% individual tensors
#def normalize_neg1_1(t):
#    """normalize input tensor t to range -1...1"""
#    return 2 * (t-tf.reduce_min(t))/tf.reduce_max(t) - 1
#
#def normalize_0_1(t):
#    """normalize input tensor t to range -1...1"""
#    return (t-tf.reduce_min(t))/tf.reduce_max(t)
#
## %% batch
##def normalize_neg1_1_batch(batch):
##    """
##    normalize all 3d-images in batch to range -1....1 assuming 5d input with
##    shape (n_batch, depth, height, width, channels)
##    """
##    return 
#def normalize_along_axis_neg1_1(t, axis):
#    """normalize input tensor t to range -1...1 along axis given"""
#    return 2 * (t-tf.reduce_min(t, axis=axis))/tf.reduce_max(t, axis=axis) - 1
#
## %% dataset (using numpy!)
#def normalize_std_mean(dataset):
#    """use numpy to run normalization on entire dataset"""
#   
#    mean = np.mean(dataset, axis=(1,2,3))
#    std = np.std(dataset, axis=(1,2,3))
#    
#    # this should be done only once on entire dataset
#    print("Mean: ", str(mean), ", Std: ", str(std))
#    
#    # uses broadcasting trick
#    normalized_dataset = (dataset - mean[:, np.newaxis, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis, np.newaxis]
#    
#    return normalized_dataset
#
##def normalize_0_1(dataset):
## use loop to normalize one after another
##    """use numpy to run normalization on entire dataset"""
##   
##    mean = np.mean(dataset, axis=(1,2,3))
##    std = np.std(dataset, axis=(1,2,3))
##    
##    # this should be done only once on entire dataset
##    print("Mean: ", str(mean), ", Std: ", str(std))
##    
##    # uses broadcasting trick
##    normalized_dataset = (dataset - mean[:, np.newaxis, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis, np.newaxis]
##    
##    return normalized_dataset



#sh = np_x.shape
#print(np.expand_dims(np.min(np_x, axis=1), axis=-1))
#print(np.min(np_x, axis=1).reshape((sh[0], 1)))
#print(np_normalize_along_axis_2d_neg1_1(np_x, 0))

#x = tf.constant(np_x)
#with tf.Session() as sess:
    #print(sess.run(normalize_along_axis_neg1_1(x, 0)))


#class MaxAbsScaler(BaseEstimator, TransformerMixin):
#    """Scale each feature by its maximum absolute value.
#
#    This estimator scales and translates each feature individually such
#    that the maximal absolute value of each feature in the
#    training set will be 1.0. It does not shift/center the data, and
#    thus does not destroy any sparsity.
#
#    This scaler can also be applied to sparse CSR or CSC matrices.
#
#    .. versionadded:: 0.17
#
#    Parameters
#    ----------
#    copy : boolean, optional, default is True
#        Set to False to perform inplace scaling and avoid a copy (if the input
#        is already a numpy array).
#
#    Attributes
#    ----------
#    scale_ : ndarray, shape (n_features,)
#        Per feature relative scaling of the data.
#
#        .. versionadded:: 0.17
#           *scale_* attribute.
#
#    max_abs_ : ndarray, shape (n_features,)
#        Per feature maximum absolute value.
#
#    n_samples_seen_ : int
#        The number of samples processed by the estimator. Will be reset on
#        new calls to fit, but increments across ``partial_fit`` calls.
#
#    See also
#    --------
#    maxabs_scale: Equivalent function without the estimator API.
#
#    Notes
#    -----
#    For a comparison of the different scalers, transformers, and normalizers,
#    see :ref:`examples/preprocessing/plot_all_scaling.py
#    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
#    """
#
#    def __init__(self, copy=True):
#        self.copy = copy
#
#    def _reset(self):
#        """Reset internal data-dependent state of the scaler, if necessary.
#
#        __init__ parameters are not touched.
#        """
#
#        # Checking one attribute is enough, becase they are all set together
#        # in partial_fit
#        if hasattr(self, 'scale_'):
#            del self.scale_
#            del self.n_samples_seen_
#            del self.max_abs_
#
#    def fit(self, X, y=None):
#        """Compute the maximum absolute value to be used for later scaling.
#
#        Parameters
#        ----------
#        X : {array-like, sparse matrix}, shape [n_samples, n_features]
#            The data used to compute the per-feature minimum and maximum
#            used for later scaling along the features axis.
#        """
#
#        # Reset internal state before fitting
#        self._reset()
#        return self.partial_fit(X, y)
#
#    def partial_fit(self, X, y=None):
#        """Online computation of max absolute value of X for later scaling.
#        All of X is processed as a single batch. This is intended for cases
#        when `fit` is not feasible due to very large number of `n_samples`
#        or because X is read from a continuous stream.
#
#        Parameters
#        ----------
#        X : {array-like, sparse matrix}, shape [n_samples, n_features]
#            The data used to compute the mean and standard deviation
#            used for later scaling along the features axis.
#
#        y : Passthrough for ``Pipeline`` compatibility.
#        """
#        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
#                        estimator=self, dtype=FLOAT_DTYPES)
#
#        if sparse.issparse(X):
#            mins, maxs = min_max_axis(X, axis=0)
#            max_abs = np.maximum(np.abs(mins), np.abs(maxs))
#        else:
#            max_abs = np.abs(X).max(axis=0)
#
#        # First pass
#        if not hasattr(self, 'n_samples_seen_'):
#            self.n_samples_seen_ = X.shape[0]
#        # Next passes
#        else:
#            max_abs = np.maximum(self.max_abs_, max_abs)
#            self.n_samples_seen_ += X.shape[0]
#
#        self.max_abs_ = max_abs
#        self.scale_ = _handle_zeros_in_scale(max_abs)
#        return self
#
#    def transform(self, X):
#        """Scale the data
#
#        Parameters
#        ----------
#        X : {array-like, sparse matrix}
#            The data that should be scaled.
#        """
#        check_is_fitted(self, 'scale_')
#        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
#                        estimator=self, dtype=FLOAT_DTYPES)
#
#        if sparse.issparse(X):
#            inplace_column_scale(X, 1.0 / self.scale_)
#        else:
#            X /= self.scale_
#        return X
#
#    def inverse_transform(self, X):
#        """Scale back the data to the original representation
#
#        Parameters
#        ----------
#        X : {array-like, sparse matrix}
#            The data that should be transformed back.
#        """
#        check_is_fitted(self, 'scale_')
#        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
#                        estimator=self, dtype=FLOAT_DTYPES)
#
#        if sparse.issparse(X):
#            inplace_column_scale(X, self.scale_)
#        else:
#            X *= self.scale_
#        return X
#
#
#def maxabs_scale(X, axis=0, copy=True):
#    """Scale each feature to the [-1, 1] range without breaking the sparsity.
#
#    This estimator scales each feature individually such
#    that the maximal absolute value of each feature in the
#    training set will be 1.0.
#
#    This scaler can also be applied to sparse CSR or CSC matrices.
#
#    Parameters
#    ----------
#    X : array-like, shape (n_samples, n_features)
#        The data.
#
#    axis : int (0 by default)
#        axis used to scale along. If 0, independently scale each feature,
#        otherwise (if 1) scale each sample.
#
#    copy : boolean, optional, default is True
#        Set to False to perform inplace scaling and avoid a copy (if the input
#        is already a numpy array).
#
#    See also
#    --------
#    MaxAbsScaler: Performs scaling to the [-1, 1] range using the``Transformer`` API
#        (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).
#
#    Notes
#    -----
#    For a comparison of the different scalers, transformers, and normalizers,
#    see :ref:`examples/preprocessing/plot_all_scaling.py
#    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
#    """  # noqa
#    # Unlike the scaler object, this function allows 1d input.
#
#    # If copy is required, it will be done inside the scaler object.
#    X = check_array(X, accept_sparse=('csr', 'csc'), copy=False,
#                    ensure_2d=False, dtype=FLOAT_DTYPES)
#    original_ndim = X.ndim
#
#    if original_ndim == 1:
#        X = X.reshape(X.shape[0], 1)
#
#    s = MaxAbsScaler(copy=copy)
#    if axis == 0:
#        X = s.fit_transform(X)
#    else:
#        X = s.fit_transform(X.T).T
#
#    if original_ndim == 1:
#        X = X.ravel()
#
#    return X


