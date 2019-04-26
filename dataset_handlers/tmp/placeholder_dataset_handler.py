#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:38:58 2018

@author: soenke
"""

import numpy as np
import h5py
import random
#import os.path as path
from decorators import deprecated

@deprecated("not compatible with code adapted to tfdata-input pipeline")
class DatasetHandler(object):
    """
    - loads dataset from h5-file into memory
    - makes sure X and Y have same shape
    - provides methods to get current_batch() and next_batch()
    """
    
    def __init__(self, filename, batch_size, random_seed=None):
        """
        Args:
        
        filename (str):  name of file containing data.  Must be h5-file with 
        fields "X" and "Y".
        batch_size (int): size of mini-batches to be created
        """
        
        self.X, self.Y = self.load_data(filename)  # np-arrays
        self.check_data_shape()  # X and Y must have same shape
        self.n_images = self.X.shape[0]
        self.batch_size = batch_size
        self.bim = BatchIndexMaker(self.n_images, self.batch_size, random_seed)
        self.n_batches_per_epoch = self.bim.n_batches
        
        self.current_X_batch = self.X[self.bim.current_index_batch]
        self.current_Y_batch = self.Y[self.bim.current_index_batch]
        
    def load_data(self, filename):
        # get file extension
        dotpos = filename[::-1].index('.')
        file_extension = filename[-dotpos:]
        # load
        if file_extension == 'h5':
            X, Y = self.load_h5(filename=filename)
        else:
            raise ValueError('only h5-files with file-extension .h5 ' +
                             'in filename are supported as of now.')
        return X, Y

    def load_h5(self, filename):
        # h5-file must have attributes X and Y
        data_file = h5py.File(filename, 'r')
        X = np.array(data_file['X'])
        Y = np.array(data_file['Y'])
        data_file.close()
        return X, Y
    
    def check_data_shape(self):
        if self.X.shape != self.Y.shape:
            msg = ("X and Y must have same shape.\n" +
                  "X has shape" + str(self.X.shape) +".\n" +
                  "Y has shape" + str(self.Y.shape) + ".")
            raise ValueError(msg)

    # TODO: remove and adapt to be runnable with new tf-data-pipeline
    def current_batch(self): 
        return self.current_X_batch, self.current_Y_batch

    def next_batch(self):
        self.current_X_batch = self.X[self.bim.next_index_batch()]
        self.current_Y_batch = self.Y[self.bim.current_index_batch]
        return self.current_X_batch, self.current_Y_batch

#    def update_current_batch(self):
#        """reset current_X_batch and current_Y_batch after preprocessing"""
#        self.current_X_batch = self.X[self.bim.current_index_batch]
#        self.current_Y_batch = self.Y[self.bim.current_index_batch]


class BatchIndexMaker(object):
    
    def __init__(self, n_images, batch_size, random_seed=None):
        """
        ensures all images are taken into account equally by starting first 
        batch of second epoch with those images not used in first epoch
        (if batch_size != integer * n_images).
        Thus have to keep track of the list of indices used to generate
        batches.  Initial indexlist is just random.
        TODO: strictly speaking the batch statistics may be screwed up
        by this a bit.
        """
        self.n_images = n_images
        self.batch_size = batch_size
        if self.batch_size > self.n_images:
            raise ValueError("batch size cannot be greater than number of " +
                             "images in the dataset")    
        self.n_batches = self.n_images // self.batch_size        
        self.counter = 0
        # TODO: the code might be more intuitive, when list of current indices
        # is used instead of indices used next.
        self.next_indexlist = list(range(self.n_images))
        random.seed(random_seed)
        random.shuffle(self.next_indexlist)
        self.batched_indices = self.create_batched_indices()
        self.current_index_batch = self.batched_indices[0] 

    def create_batched_indices(self):
        """create index chunks corresponding to batches."""
        batched_ids = [self.next_indexlist[i:i+self.batch_size] for i in range(0, self.n_images, self.batch_size)]
        if len(batched_ids[-1]) != self.batch_size:
            omitted_chunk = batched_ids.pop()  # pops in place
            sublist = self.next_indexlist[0:-len(omitted_chunk)]
            random.shuffle(sublist)
            # new indexlist
            self.next_indexlist = omitted_chunk + sublist
        else:
            # new indexlist
            random.shuffle(self.next_indexlist)  # shuffles in place
        return batched_ids

    def next_index_batch(self):
        self.counter += 1
        if self.counter == self.n_batches:
            self.batched_indices = self.create_batched_indices()
            self.counter = 0
        self.current_index_batch = self.batched_indices[self.counter]
        return self.current_index_batch




#def _subtract_mean(X, mean=None):
#    if mean is None:
#        mean = np.mean(X)
#    return X - mean
#
#def _normalize_scale(X, std=None):
#    if std is None:
#        std = np.std(X)
#    return X / std


def main():
    # test batch_index maker.
#    bim = BatchIndexMaker(8,3)
#    print("n_images:", bim.n_images)
#    print("batch_size:", bim.batch_size)
#    print("n_batches:", bim.n_batches)
#    print("counter:", bim.counter)
#    print("batched_indices:", bim.batched_indices)
#    print("next_indexlist:", bim.next_indexlist)
#    print("counter:", bim.counter)
#    print("current_index_batch:", bim.current_index_batch)
#    bim.next_index_batch()
#    print("counter:", bim.counter)
#    print("current_index_batch:", bim.current_index_batch)
#    bim.next_index_batch()
#    print("batched_indices:", bim.batched_indices)
#    print("next_indexlist:", bim.next_indexlist)
#    print("counter:", bim.counter)
#    print("current_index_batch:", bim.current_index_batch)
    
    # test dataset handler
    import os.path as path
    dataset = DatasetHandler("/home/soenke/Pictures/datasets/modified/" +
                             "March_2013_VascuSynth_Dataset/" + 
                             "h5_deconv_dataset/poisson/num_photons10_bgr10/" +
                             "na1.2_ri1.33_scaleXY2000_scaleZ5000/" + 
                             "vascu_pairs_train.h5", 10)
#            path.join("/Pictures", "datasets", "modified", 
#                                       "March_2013_VascuSynth_Dataset",
#                                       "h5_deconv_dataset", "poisson", 
#                                       "num_photons10_bgr10", 
#                                       "na1.2_ri1.33_scaleXY2000_scaleZ5000",
#                                       "vascu_pairs_train.h5"), 10)
    print("n_images:", dataset.n_images)
    print("batch_size:", dataset.batch_size)
    print("batch", dataset.bim.counter+1, "/", dataset.bim.n_batches)
    print("indices", dataset.bim.current_index_batch)
    print("X_batch.shape:", dataset.current_X_batch.shape)
    print("Y_batch.shape:", dataset.current_Y_batch.shape)
    dataset.next_batch()
    print("batch", dataset.bim.counter+1, "/", dataset.bim.n_batches)
    print("indices", dataset.bim.current_index_batch)
    print("X_batch.shape:", dataset.current_X_batch.shape)
    print("Y_batch.shape:", dataset.current_Y_batch.shape)
    dataset.next_batch()
    print("batch", dataset.bim.counter+1, "/", dataset.bim.n_batches)
    print("indices", dataset.bim.current_index_batch)
    print("X_batch.shape:", dataset.current_X_batch.shape)
    print("Y_batch.shape:", dataset.current_Y_batch.shape)

    print("before preprocessing:")
    print("min(X):", dataset.X.min())
    print("max(X):", dataset.X.max())
    print("max(X_batch):", dataset.current_X_batch.max())
    print("max(Y_batch):", dataset.current_Y_batch.max())
    mean, std = dataset.preprocess()
    print("mean:", mean)
    print("std:", std)
    print("after preprocessing: ")
    print("mean:", np.mean(dataset.X))
    print("std:", np.std(dataset.X))
    print("min(X):", dataset.X.min())
    print("max(X):", dataset.X.max())
    print("max(X_batch):", dataset.current_X_batch.max())
    print("max(Y_batch):", dataset.current_Y_batch.max())
    
    print("batch", dataset.bim.counter+1, "/", dataset.bim.n_batches)
    print("indices", dataset.bim.current_index_batch)
    print("X_batch.shape:", dataset.current_X_batch.shape)
    print("Y_batch.shape:", dataset.current_Y_batch.shape)
    dataset.next_batch()
    print("batch", dataset.bim.counter+1, "/", dataset.bim.n_batches)
    print("indices", dataset.bim.current_index_batch)
    print("X_batch.shape:", dataset.current_X_batch.shape)
    print("Y_batch.shape:", dataset.current_Y_batch.shape)

if __name__ == "__main__":
    main()