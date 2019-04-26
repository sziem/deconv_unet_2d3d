#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:38:01 2018

@author: soenke
"""

# own imports
from . import dataset_utils as dutils
#from decorators import deprecated

import json
import os
from warnings import warn
import tensorflow as tf
from numpy import float32 as np_float32

# I recommend using a handler based on BaseListDatasetHandler (!)

# TODO: currently it will repeat the exact same sequence when loading from 
#       checkpoint with same random seed.  Is it possible to store iterator
#       state in some way? -> workaround: continue training with different
#       random seed.
# TODO: make sure these are done on CPU
#       with tf.name_scope("load_data"):  #with tf.device('/cpu:0'):
# TODO: test dataset handlers other than listdatasethandler
# TODO: each dataset will create 2 unconnected placeholders for shape determination
#       in tensorboard.  Hide these somewhere, eg. in name scope

# %% Base dataset handler

# TODO: make sure dataset and data_format match
class BaseDatasetHandler(object):
    """
    provides methods to manipulate dataset.
    """
    def __init__(self, dataset, x_shape=None, y_shape=None, n_images=None, 
                 mean=None, std=None, dataset_id=None, 
                 data_format=None):
        """
        minimal init which accepts an initialized tf-data-dataset
        """
        self.dataset = dataset
        self.x_shape = x_shape #self.dataset.batch(2).output_shapes[0]
        self.y_shape = y_shape #self.dataset.batch(2).output_shapes[1]
        self.n_images = n_images # cannot be easily inferred from tf-dataset
        self.mean = mean
        self.std = std
        self.dataset_id = self._set_dataset_id(dataset_id)
        self.data_format = self._set_data_format(data_format)
        self.handler_type = "base"

    def shuffle_and_repeat(self, buffer_size, n_epochs, random_seed=None,
                           fused=False):
        """
        - provides an operation to shuffle dataset before each epoch
          (or subsets of size buffer_size of the dataset, if dataset is larger
          than buffer_size).  Shuffling is done according to random_seed.
        - repeats shuffled dataset n_epochs times
        - if fused, it will do both in a single operation.  This has thrown
          errors in the past, but might be fixed in newer tf-versions.
        """
        # fused (faster, especially for large batch sizes)
        if fused:  # TODO: update to tf 1.12
            self.dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat(
                    buffer_size, count=n_epochs, seed=random_seed))
        else:
            self.dataset = self.dataset.shuffle(buffer_size, seed=random_seed)
            self.dataset = self.dataset.repeat(n_epochs)
        return self.dataset
    
    def _repeat(self, n_epochs):
        """
        Only repeat without shuffling.  
        Useful for Testing or Validation
        """
        self.dataset = self.dataset.repeat(n_epochs)
        return self.dataset
    
    def map_and_batch(self, map_fn, batch_size, num_parallel_calls=None, 
                      fused=False):
        """
        Maps data pairs, e.g. for data preprocessing and forms minibatches.
        
        - map_fn must contain a mapping for both x and y
        (take arguments x and y and return x_mapped and y_mapped)
        
        - batch_size (int) : size of resulting minibatches.  Note that batch
          dimension will be set to 'None' by tf.  This is because it might not
          be possible to split the dataset into batches. In that case, tf will
          make the last batch smaller by default.
        
        - num_parallel calls is num_parallel_calls for tf.data.Dataset.map()
          setting of num_parallel_calls depends on hardware, training data, 
          cost of map_fn and other processing done on Cpu.
          ideally this would be number of available cpus, but I think I'll put 
          less to allow cpus for other tasks too.
          12 would be number of physical cores on cpu of fa8-titanx
        
        - if fused, it will do both in a single operation.  This has thrown
          errors in the past, but might be fixed in newer tf-versions.
        """
        # fused (faster, especially for large batch sizes)
        if fused:  # TODO: update to tf 1.12
            # note that num parallel calls does not necessarily have to be set
            self.dataset = self.dataset.apply(tf.contrib.data.map_and_batch(
              map_func=map_fn, batch_size=batch_size, 
              num_parallel_calls=num_parallel_calls))
        else:
            self.dataset = self.dataset.map(map_fn, 
              num_parallel_calls=num_parallel_calls)
            self.dataset = self.dataset.batch(batch_size)
        return self.dataset

    def cache(self):
        """
        Caches the dataset in memory.
        
        From performance guide (tf 1.9)
        https://www.tensorflow.org/performance/datasets_performance
        If the user-defined function passed into the map transformation is 
        expensive, apply the cache transformation after the map transformation 
        as long as the resulting dataset can still fit into memory 
        or local storage
        """
        self.dataset = self.dataset.cache()
        return self.dataset
            
    def prefetch(self, buffer_size):
        self.dataset = self.dataset.prefetch(buffer_size)
        return self.dataset
    
    def make_iterator(self):
        self.iterator = self.dataset.make_one_shot_iterator()
    
    def get_output_shapes(self):
        return self.dataset.output_shapes
    
    def get_x_channels(self):
        return self.x_shape.as_list()[self._get_channel_axis()]
    
    def get_y_channels(self):
        return self.y_shape.as_list()[self._get_channel_axis()]
    
    def _get_channel_axis(self):
        if self.data_format == "channels_last":
            return -1
        elif self.data_format == "channels_first":
            return 1
        else:
            raise ValueError("unknown data_format: " + self.data_format + 
                             ". " + "Use 'channels_first' or 'channels_last'.")
    
    def next_batch(self):
        try:
            return self.iterator.get_next()
        except Exception as e:
            raise e
        # TODO: except iterator is not defined
    
    def _set_dataset_id(self, dataset_id=None):
        if dataset_id is None:
            warn("dataset_id is not set.  dataset_id can be used to " +
                 "identify the dataset that is used to train the model. " +
                 "Will use default 'unknown_dataset'.")
            # TODO: or just leave None?
            dataset_id = "unknown_dataset"
        self.dataset_id = dataset_id
        return self.dataset_id
    
    def _set_data_format(self, data_format):
        if data_format is None:
            warn("data_format is None.  Assuming default 'channels_last', "
                 "but you may want to set this differently.")
            self.data_format = "channels_last"
        elif data_format in ["channels_last", "channels_first"]:
            self.data_format = data_format
        else:
            raise ValueError("Unknown data_format " + data_format)
        return self.data_format

# %% BaseNumpyDatasetHandler: Base for datasets that fit into memory
#@deprecated("not up to date. Behavior may differ from ListDatasetHandler")
class BaseNumpyDatasetHandler(BaseDatasetHandler):
    """
    Base Dataset Handler for the case, where dataset exists as a numpy array 
    in memory.  See note, why this should be handled differently.
    
    Inherits from BaseDatasetHandler.
    
    NOTE:
    Docs (tf 1.9) recommend using a workaround for input of datasets from
    data like numpy arrays.  This is because the entire numpy array would be 
    embedded as a tf.constant in the graph object, if it were passed directly
    to tf-data.  This is wasteful in storing checkpoints on one hand and 
    might exceed tf-internal limits for the graph size on the other hand.  
    
    See the following links for more information:
    https://www.tensorflow.org/versions/r1.9/guide/datasets#consuming_numpy_arrays
    https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/data/Dataset
    #from_tensor_slices    

    (when using another version of tf, check with the appropriate guides, if 
    this is still the suggested way)
    """
    def __init__(self, np_X, np_Y, mean=None, std=None, dataset_id=None,
                 data_format=None):
        # workaround via placeholder as described in class docstring:
        # Note that entire dataset is provided during initialization.  
        # Thus, there is no speed loss due to the use of a placeholder.
        X_plh = tf.placeholder(tf.float32, shape=np_X.shape)  # do not force float32?
        Y_plh = tf.placeholder(tf.float32, shape=np_Y.shape)
        self.iterator_feed_dict = {X_plh:np_X, Y_plh:np_Y}
        # TODO: consider joining dataset together at a later stage 
        # after mapping -> give two map functions (yes!)
        dataset = tf.data.Dataset.from_tensor_slices((X_plh, Y_plh))
        
        # set attributes
        super().__init__(dataset,
             x_shape=self.dataset.batch(2).output_shapes[0],
             y_shape=self.dataset.batch(2).output_shapes[1],
             n_images=np_X.shape[0],
             mean=mean,
             std=std,
             dataset_id=dataset_id,
             data_format=data_format)
        self.handler_type = "np"
        
    # Override the one provided in base
    def make_iterator(self):
        raise RuntimeError("You need to run make_iterator_and_initialize to " +
                           "use NumpyDatasetHandler.")
    
    def make_iterator_and_initialize(self, sess):
        self.iterator = self.dataset.make_initializable_iterator()

        sess.run(self.iterator.initializer, feed_dict=self.iterator_feed_dict)
        return self.iterator
    # TODO: how to make more explicit, which methods are defined in Base...?


# %% BaseListDatasetHandler: Base for datasets managed from filelist (preferred)
# This is scalable to large datasets and versatile --> currently preferred

# TODO: change to use only load_fn for a single img
# -> pair can then be loaded by calling it twice
# -> that way, Unsupervised Handler could be implemented much easier
class BaseListDatasetHandler(BaseDatasetHandler):
    """
    Maintains Dataset in the form of a list.  
    Loads images later via map_and_batch!
    """
    def __init__(self, x_filelist, y_filelist, load_pair_fn, 
                 mean=None, std=None, dataset_id=None, 
                 data_format=None):
        """
        x_filelist and y_filelist should contain paths to the files forming the 
        image pairs forming the dataset.  Both lists must have the same length.
        load_pairs_fn should be function taking 2 arguments:
            - full paths to x- and y-files (including file extensions)
        and returning 2 numpy arrays (loaded x and loaded y).
        
        TODO: docstring -> parts of docstring should be the same for all Bases
        """
        if len(x_filelist) != len(y_filelist):
            raise ValueError(
                    "x_filelist and y_filelist must have the same length.")
        
        # define dataset
        dataset = tf.data.Dataset.from_tensor_slices(
                (tf.constant(x_filelist), tf.constant(y_filelist)))
        
        # set attributes
        if not x_filelist:  # empty
            warn("x_filelist is empty.")
            x_shape = (0,)
            y_shape = (0,)
        else:
            x_shape = load_pair_fn(x_filelist[0], y_filelist[0])[0].shape
            y_shape = load_pair_fn(x_filelist[0], y_filelist[0])[1].shape
        # (make first shape dimension "None")
        x_tmp = tf.placeholder(tf.float32, [None,] + list(x_shape))
        y_tmp = tf.placeholder(tf.float32, [None,] + list(y_shape))
        x_batch_shape = x_tmp.shape
        y_batch_shape = y_tmp.shape
        super().__init__(dataset,
             x_shape=x_batch_shape,
             y_shape=y_batch_shape,
             n_images=len(x_filelist),
             mean=mean,
             std=std,
             dataset_id=dataset_id,
             data_format=data_format)
        self._np_load_pair_fn = load_pair_fn
        self.handler_type = "list"
    
    def _np_load_pair(self, x_path, y_path):
        return self._np_load_pair_fn(
                str(x_path, encoding='utf-8'),
                str(y_path, encoding='utf-8'))
    
    # overriding map_and_batch
    def map_and_batch(self, map_fn, batch_size, num_parallel_calls=None, 
                      fused=False):
        """
        overriding map_and_batch.  
        
        - map_fn must contain a mapping for both x and y
        (take arguments x and y and return x_mapped and y_mapped)
        
        - num_parallel calls is num_parallel_calls for tf.data.Dataset.map()
        
        setting of num_parallel_calls depends on hardware, training data, 
        cost of map_fn and other processing done on Cpu.
        ideally this would be number of available cpus, but I think I'll put 
        less to allow cpus for other tasks too.
        
        12 would be half of available cores on cpu of fa8-titanx
        """
        # I couldn't manage to combine them, so this will impact performance
#        map_fn = lambda file_path: map_fn(tf.py_func(
#                _load_pair, [file_path,], [tf.float32, tf.float32]))
        
        def _tf_load_pair(x_path, y_path):
            x, y = tf.py_func(
                    self._np_load_pair, [x_path, y_path], 
                    [tf.float32, tf.float32])
            x.set_shape(self.x_shape[1:])
            y.set_shape(self.y_shape[1:])
            return x, y
        
        # decode, then crop, then batch  --> saves memory 
        # crop and batch can be fused, but not with decode
        self.dataset = self.dataset.map(_tf_load_pair)
        self.dataset = super().map_and_batch(map_fn, batch_size, 
                            num_parallel_calls=num_parallel_calls, fused=fused)
        return self.dataset

# %% BaseGeneratorDatasetHandler: Base for datasets returned from generator

# TODO: untested!
class _BaseGeneratorDatasetHandler(BaseDatasetHandler):
    """
    loads data into tf-data-dataset using a generator.
    Avoid using this if you want to shuffle your data as it may be more effort.
    """
    def __init__(self, x_gen, y_gen, x_shape, y_shape, n_images,
                 mean=None, std=None, dataset_id=None,
                 data_format=None):
        
        # afaik this would consume first value of generator
        # x_shape=next(x_gen).shape
        # y_shape=next(y_gen).shape
        # n_images = sum(1 for _ in x_gen)

        # TODO: consider joining dataset together at a later stage 
        # after mapping and give two map functions (yes!)
        # It is assumed that x and y have the same shape
        X_dataset = tf.data.Dataset.from_generator(
                lambda: x_gen,
                output_types=tf.float32, 
                output_shapes=x_shape)
        Y_dataset = tf.data.Dataset.from_generator( 
                lambda: y_gen,
                output_types=tf.float32,
                output_shapes=y_shape)
        dataset = tf.data.Dataset.zip((X_dataset, Y_dataset))
        
        # set attributes
        super().__init__(dataset,
             x_shape=x_shape,
             y_shape=y_shape,
             n_images=n_images, 
             mean=mean,
             std=std,
             dataset_id=dataset_id,
             data_format=data_format)
        self.handler_type = "gen"    
    
    def shuffle_and_repeat(self, buffer_size, n_epochs, random_seed=None,
                           fused=False):
        # shuffling a generator without first loading it completely is
        # not possible.
        raise RuntimeError(
                "BaseGeneratorDatasetHandler does not support shuffling. " +
                "You need to implement shuffling during data generation, if " +
                "necessary. Use 'repeat' instead of 'shuffle_and_repeat' here.")
    
    def repeat(self, n_epochs):
        return self._repeat(n_epochs) 


# %% Custom Dataset Handlers built on top of BaseListDatasetHandler

# View this as an example on how to use BaseListDatasetHandler
"""
docstring: TODO
"""
class TwoFoldersListDatasetHandler(BaseListDatasetHandler): 
    def __init__(self, x_folder, y_folder, load_pair_fn, 
                 mean=None, std=None, dataset_id=None, 
                 data_format=None):
        """
        docstring: TODO
        assumes all x-files are in one folder
        assumes all y-files are in one folder
        (y_folder should be different unless both x and y point to the same 
         files, e.g. mat-files containing two variables)
        (x_folder and y_folder should contain no other files than the dataset!)
        load_pair_fn should be function taking 2 arguments:
            - full paths to x- and y-files (including file extensions)
        and returning 2 numpy arrays (loaded x and loaded y).
        """
        x_filelist = [os.path.join(x_folder,f) for f in os.listdir(x_folder) 
                      if os.path.isfile(os.path.join(x_folder, f))]
        y_filelist = [os.path.join(y_folder,f) for f in os.listdir(y_folder) 
                      if os.path.isfile(os.path.join(y_folder, f))]
        # previously also had these arguments
        # (ftype should be string, eg. 'mat'
        # x_ftype, y_ftype
        # if f.endswith(y_ftype) and os.path.isfile(...))
        super().__init__(
                x_filelist,
                y_filelist, 
                load_pair_fn,
                mean=mean, 
                std=std, 
                dataset_id=dataset_id, 
                data_format=data_format)

"""
docstring: TODO
"""    
class ManyFoldersListDatasetHandler(BaseListDatasetHandler):
    def __init__(self, dataset_path, x_name, y_name, load_pair_fn, 
                 mean=None, std=None, dataset_id=None, 
                 data_format=None):
        """
        docstring: TODO
        assumes each data pair lies together in its own folder in dataset_path.
        Dataset_path should contain no other folders than those containing 
        data pairs.
        x_name and y_name are the names of the files in the folder.
        load_pair_fn should be function taking 2 arguments:
            - full paths to x- and y-files (including file extensions)
        and returning 2 numpy arrays (loaded x and loaded y).
        """
        x_filelist = [os.path.join(f, x_name) 
                      for f in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, f))]
        y_filelist = [os.path.join(f, y_name) 
                      for f in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, f))]
        super().__init__(
                x_filelist,
                y_filelist, 
                load_pair_fn,
                mean=mean, 
                std=std, 
                dataset_id=dataset_id, 
                data_format=data_format)


# %% Custom Dataset Handlers built on top of BaseNumpyDatasetHandler
# Note that I don't recommend this.  Generating a h5 file might be wasteful.
class H5NumpyDatasetHandler(BaseNumpyDatasetHandler):
    """
    loads entire dataset from h5-file into memory as np-array and
    then converts it to tf-data-dataset.  
    Inherits from NumpyDatasetHandler (-> see there for documentation)
    """
    def __init__(self, dataset_path, x_name, y_name, dataset_id=None, 
                 data_format=None, cal_mean=True, cal_std=True):
        """
        Args:
            filename (str) :  name of file containing data.  Must be h5-file 
            with fields x_name and y_name
            docstring TODO
        """
        # Load entire h5-dataset as np-arrays
        np_X = dutils._load_h5(dataset_path, field=x_name)
        np_Y = dutils._load_h5(dataset_path, field=y_name)
        mean = None
        std = None
        if cal_mean:
            mean = np_X.mean()
        if cal_std:
            std = np_Y.std()
        super().__init__(
                np_X, 
                np_Y, 
                mean=mean, 
                std=std, 
                dataset_id=dataset_id,
                data_format=data_format)

# Try if you have existing ListDatasetHandler and it fits in memory.
# There is potential speed-up of a numpy dataset handler due to caching 
# (if caching at the right place)

# -> can be easily combined with handlers built on top of 
#    ListDatasetHandler
# code very similar to H5NumpyDatasetHandler
class ListNumpyDatasetHandler(BaseNumpyDatasetHandler):
    """
    Load data as given in list, but put in np-array to allow caching for 
    potential speed up
    """
    def __init__(self, x_filelist, y_filelist, load_fn, dataset_id=None, 
                 data_format=None, cal_mean=True, cal_std=True):
        # Load dataset as np-array
        np_X = dutils._load_filelist(x_filelist, load_fn)
        np_Y = dutils._load_filelist(y_filelist, load_fn)
        mean = None
        std = None
        if cal_mean:
            mean = np_X.mean()
        if cal_std:
            std = np_Y.std()
        super().__init__(
                np_X, 
                np_Y, 
                mean=mean, 
                std=std, 
                cal_mean=cal_mean, 
                cal_std=cal_std, 
                dataset_id=dataset_id)


# %% Dataset Handler specific to my work
# also manages divsion into train/validation/test-folds
# but has a very specific format.
class VascuPairsListDatasetHandler(BaseListDatasetHandler):
    """
    Uses file structure returned by vascu_synth.  Each data pair resides in 
    its own folder named "image0", "image1", etc. in dataset_path
    Uses dataset_specs file to get filelists and properties.  This can be 
    generated by generate_dataset_specs.write_dataset_specs(...)
    """
    def __init__(self, dataset_path, json_file="dataset_specs.json", 
                 mode="train", data_format="channels_last"):
        """
        related to ManyFoldersListDatasetHandler, but
        uses indexlist from dataset_specs
        """
        # get dataset_specs
        with open(os.path.join(dataset_path, json_file), 'r') as file:
            d = file.read()
        dataset_specs = json.loads(d)
        
        # get filelists
        indexlist = dataset_specs[mode+"_images"]
        
        x_name = dataset_specs["x_name"]
        y_name = dataset_specs["y_name"]
        x_filelist = [os.path.join(dataset_path, "image" + str(i), x_name)
                      for i in indexlist]
        y_filelist = [os.path.join(dataset_path, "image" + str(i), y_name)
                      for i in indexlist]
        
        # define load function for the b/w mat-files and add channel dim
        # TODO: use dataset_specs["x_format"] and dataset_specs["y_format"]
        # together with data_format to determine where to add dims.
        # TODO change data_format to format string (ndhwc etc)
        if data_format == "channels_last":
            channel_axis = -1
        elif data_format == "channels_first":
            channel_axis = 0
        else:
            raise ValueError(
                    "data_format " + data_format + " is not " +
                    "supported. Please use either channels_first or " +
                    "channels_last.")
        def load_pair_fn(x_path, y_path):
            np_x = dutils.np_load(
                    x_path, expand_dims=channel_axis, version='7'
                    ).astype(np_float32)
            np_y = dutils.np_load(
                    y_path, expand_dims=channel_axis, version='7'
                    ).astype(np_float32)
            return np_x, np_y
        
        # define dataset
        super().__init__(
                x_filelist,
                y_filelist, 
                load_pair_fn,
                mean=dataset_specs["training_mean"], 
                std=dataset_specs["training_std"], 
                dataset_id=dataset_specs["dataset_id"], 
                data_format=data_format)

# TODO: much code is copied.  Maybe it is better to use generate different 
# dataset specs for Unsupervised and reuse VascuPairsListDatasetHandler?
class UnsupervisedVascuPairsListDatasetHandler(VascuPairsListDatasetHandler):
    """
    related to VascuPairsListDatasetHandler, but erases all traces of y
    """
    def __init__(self, dataset_path, json_file="dataset_specs.json", 
                 mode="train", data_format="channels_last"):
        # get dataset_specs
        with open(os.path.join(dataset_path, json_file), 'r') as file:
            d = file.read()
        dataset_specs = json.loads(d)
        
        # get filelists -> related to ManyFoldersListDatasetHandler, but
        # also uses indexlist from dataset_specs
        indexlist = dataset_specs[mode+"_images"]
        x_name = dataset_specs["x_name"]
        x_filelist = [os.path.join(dataset_path, "image" + str(i), x_name)
                      for i in indexlist]
        
        # define load function for the b/w mat-files and add channel dim
        # TODO: use dataset_specs["x_format"]
        # together with data_format to determine where to add dims.
        # TODO change data_format to format string (ndhwc etc)
        if data_format == "channels_last":
            channel_axis = -1
        elif data_format == "channels_first":
            channel_axis = 0
        else:
            raise ValueError(
                    "data_format " + data_format + " is not " +
                    "supported. Please use either channels_first or " +
                    "channels_last.")
        def load_pair_fn(x_path, dummy):  # needs to be function of 2 args
            np_x = dutils.np_load(
                    x_path, expand_dims=channel_axis, version='7'
                    ).astype(float)
            return np_x, np_x.copy()
        
        # define dataset
        super().__init__(
                x_filelist,
                x_filelist, 
                load_pair_fn,
                mean=dataset_specs["training_mean"], 
                std=dataset_specs["training_std"], 
                dataset_id=dataset_specs["dataset_id"], 
                data_format=data_format)

# TODO: code is similar to the above two
class VascuPairsListNumpyDatasetHandler(ListNumpyDatasetHandler):
    """
    Uses file structure returned by vascu_synth.  Each data pair resides in 
    its own folder named "image0", "image1", etc. in dataset_path
    Uses dataset_specs file to get filelists and properties.  This can be 
    generated by generate_dataset_specs.write_dataset_specs(...)
    """
    def __init__(self, dataset_path, json_file="dataset_specs.json", 
                 mode="train", data_format="channels_last"):
        """
        related to ManyFoldersListDatasetHandler, but
        uses indexlist from dataset_specs
        """
        # get dataset_specs
        with open(os.path.join(dataset_path, json_file), 'r') as file:
            d = file.read()
        dataset_specs = json.loads(d)
        
        # get filelists
        indexlist = dataset_specs[mode+"_images"]
        x_name = dataset_specs["x_name"]
        y_name = dataset_specs["y_name"]
        x_filelist = [os.path.join(dataset_path, "image" + str(i), x_name)
                      for i in indexlist]
        y_filelist = [os.path.join(dataset_path, "image" + str(i), y_name)
                      for i in indexlist]
        
        # define load function for the b/w mat-files and add channel dim
        # TODO: use dataset_specs["x_format"] and dataset_specs["y_format"]
        # together with data_format to determine where to add dims.
        # TODO change data_format to format string (ndhwc etc)
        if data_format == "channels_last":
            channel_axis = -1
        elif data_format == "channels_first":
            channel_axis = 0
        else:
            raise ValueError(
                    "data_format " + data_format + " is not " +
                    "supported. Please use either channels_first or " +
                    "channels_last.")
        def load_fn(file_path):
            np_arr = dutils.np_load(file_path, expand_dims=channel_axis)
            return np_arr
        
        # define dataset
        super().__init__(
                x_filelist, 
                y_filelist, 
                load_fn, 
                dataset_id=None, 
                data_format=data_format,
                cal_mean=True, 
                cal_std=True)

        if self.x_shape != dataset_specs["x_shape"][1:]:
            warn("x_shape infered from dataset handler and x_shape " +
                 "from dataset_specs do not match")
        if self.y_shape != dataset_specs["y_shape"][1:]:
            warn("y_shape infered from dataset handler and y_shape " +
                 "from dataset_specs do not match")  

class UKJDatasetHandler(BaseDatasetHandler):
    def __init__(self, dataset_path, mean=None, std=None, dataset_id=None, 
                 data_format="channels_last"):
        """be careful with this.  Read comment in code"""
        # TODO: important note:  This relies on the fact that flattened
        # labellist corresponds to the order in which images are loaded.
        # This is only the case if x_filelist is ordered the same way
        # as indexlist.
        # an example of where this goes wrong is naming the images
        # patch0_0, patch1_0, patch2_0, ..., patch10_0, patch11_0, ...
        # Here, filelist will be [patch0_0, patch1_0, !!PATCH10_0!!, !!PATCH11_0!! , ...]
        # whereas labellist will follow the order of the patches above.

        # set path
        x_folder = os.path.join(dataset_path, "images")
        x_filelist = [os.path.join(x_folder,f) for f in os.listdir(x_folder) 
                      if os.path.isfile(os.path.join(x_folder, f))]
        
        labels_file = os.path.join(dataset_path, "labels.txt")
        labels = dutils.loadtxt(labels_file).astype(bool)  # binary classification
        
        # flatten labellist (careful; see note about ordering above)
        labellist = list(labels.reshape(1,-1)[0])
        if data_format == "channels_last":
            ch_axis = -1
        elif data_format == "channels_first":
            ch_axis = 1
        else:
            raise ValueError("Unknown data_format " + data_format)
        
        def load_pair_fn(x_path, label):
            np_x = dutils.np_load(x_path).astype(np_float32)
            np_y = dutils.label_to_image(label, np_x.shape[:-1], np_x.dtype)
            np_y = dutils.one_hot(np_y, 2, axis=ch_axis)  # 2 means binary classification
            return np_x, np_y
        
        if len(x_filelist) != len(labellist):
            raise ValueError(
                    "x_filelist and y_filelist must have the same length.")
        
        # define dataset
        dataset = tf.data.Dataset.from_tensor_slices(
                (tf.constant(x_filelist), tf.constant(labellist)))
        
        # set attributes
        if not x_filelist:  # empty
            warn("x_filelist is empty.")
            x_shape = (0,)
            y_shape = (0,)
        else:
            x_shape = load_pair_fn(x_filelist[0], labellist[0])[0].shape
            y_shape = load_pair_fn(x_filelist[0], labellist[0])[1].shape
        # (make first shape dimension "None")
        x_tmp = tf.placeholder(tf.float32, [None,] + list(x_shape))
        y_tmp = tf.placeholder(tf.float32, [None,] + list(y_shape))
        x_batch_shape = x_tmp.shape
        y_batch_shape = y_tmp.shape
        
        super().__init__(dataset,
             x_shape=x_batch_shape,
             y_shape=y_batch_shape,
             n_images=len(x_filelist),
             mean=mean,
             std=std,
             dataset_id=dataset_id,
             data_format=data_format)
        
        self._np_load_pair_fn = load_pair_fn
        self.handler_type = "list"
    
    def _np_load_pair(self, x_path, label):
        return self._np_load_pair_fn(
                str(x_path, encoding='utf-8'),
                label)
    
    # overriding map_and_batch
    def map_and_batch(self, map_fn, batch_size, num_parallel_calls=None, 
                      fused=False):
        """
        overriding map_and_batch.  
        
        - map_fn must contain a mapping for both x and y
        (take arguments x and y and return x_mapped and y_mapped)
        
        - num_parallel calls is num_parallel_calls for tf.data.Dataset.map()
        
        setting of num_parallel_calls depends on hardware, training data, 
        cost of map_fn and other processing done on Cpu.
        ideally this would be number of available cpus, but I think I'll put 
        less to allow cpus for other tasks too.
        
        12 would be half of available cores on cpu of fa8-titanx
        """
        # I couldn't manage to combine them, so this will impact performance
#        map_fn = lambda file_path: map_fn(tf.py_func(
#                _load_pair, [file_path,], [tf.float32, tf.float32]))
        
        def _tf_load_pair(x_path, label):
            x, y = tf.py_func(
                    self._np_load_pair, [x_path, label], 
                    [tf.float32, tf.float32])
            x.set_shape(self.x_shape[1:])
            y.set_shape(self.y_shape[1:])
            return x, y
        
        # decode, then crop, then batch  --> saves memory 
        # crop and batch can be fused, but not with decode
        self.dataset = self.dataset.map(_tf_load_pair)
        self.dataset = super().map_and_batch(map_fn, batch_size, 
                            num_parallel_calls=num_parallel_calls, fused=fused)
        return self.dataset


def main():
    pass

if __name__ == "__main__":
    main()