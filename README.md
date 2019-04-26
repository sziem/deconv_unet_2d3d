# NDNet
NDNet is a wrapper for running unet-like architectures in 2d and 3d for various image processing tasks, where the input and output are both pixels on the same scale.

It was originally developed for deconvolution (denoising and deblurring) of three-dimensional images.  Later, a two-dimensional unet was added as well as loss functions for semantic segmentation.

The main file is ndnet.py.  This file contains a runnable example at the bottom.

ndnet.py creates a high level network object that is responsible for train, run_on_image and test.  It will provide occasional outputs to both the console and to tensorboard.  This class also sets up all methods shared by the three modes (train, run_on_image, test) such as preprocessing.

Data is loaded using a dataset handler object.  This basically a tf-data Dataset, where some tasks are standardized.  The recommended way to load your images is to 
0. provide your images in a supported file structure
1. write a function that loads a single image 
2. pass the path to the dataset and that function to the appropriate dataset handler.

Note that not all input image sizes are allowed (depending on the convolution mode "valid" or "same").  Especially for "valid" mode, input images must have a minimum number of pixels in each direction.  That number grows exponentially with network size.  See the ndnet-presentation for examples of what may go wrong.

The network is implemented in another separate object.  The number of dimensions od conv_size and pool_size must match that of the image.  Only unet is supported right now, although there is a plan to make the network class more general.
