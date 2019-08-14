import numpy as np
import tensorflow as tf
if __name__ == '__main__':
    import ops
    import utils
else:
    from . import ops
    from . import utils
from warnings import warn

# for tensorflow 1.14 use this to avoid warnings:
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

## contains unet model object and methods for determination of shapes in the net

## BUILDING UNET
# input layer contains: conv (no batch norm or dropout)
# down-blocks contain: conv-block, pool
# up-blocks contain: (up-)layer, concat, conv-layer, conv-layer
# output layer contatins: conv

# conv_layer contains: batch_norm, conv, then activation (with relu)


# TODO: add higher layer.  Make BaseUnet that takes block_def, layer_def and 
#       ops as args
# TODO: remove print_info and give unet "reuse" attribute that will disable 
#       these prints
class Unet_v3:
    # %% init
    def __init__(self, network_depth,
            use_batch_renorm=False, use_batch_norm=False, 
            padding="same", nonlinearity=tf.nn.relu, net_channel_growth=1,
            initial_channel_growth=32, channel_growth=2,
            conv_size=(3,3,3), pool_size=(2,2,2),
            last_layer_batch_norm=None,
            data_format="channels_last", input_output_skip=False):
        """
        last_layer_batch_norm=None means True if use_bn or use_brn else False
        """

        # set params
        self.network_depth = network_depth
        #
        self.use_batch_renorm = use_batch_renorm
        self.use_batch_norm = use_batch_norm
        self.use_dropout = True  # switch on or off with dropout_rate
        self.dropout_rate = None # TODO: only needed in train. Make arg to training fxns instead?
        #
        self.padding = padding
        self.nonlinearity = nonlinearity #+ self._set_nonlinearity(nonlinearity)  
        self.weight_init = None  # TODO: only needed in train. Make arg to training fxns instead?
        #
        # initial_channel_growth defines the factor by which the channels 
        # dimension grows after the input block
        self.initial_channel_growth = initial_channel_growth
        # channel_growth defines the factor by which the channels dimension
        # grows after an encoder block (except input).
        # The same number is used to shrink in the decoder path.
        self.channel_growth = channel_growth
        # net_channel_growth (undocumented feature) can be used to have an output
        # with more channels than the input.
        # For deconv input_channels should match output_channels.  
        # A black and white input should yield a black and white output.
        # In general, eg. coloration, they could be different.  Change 
        # net_channel_growth in that case.
        # Careful!  I never tested anything but black and white in- and output        
        self.net_channel_growth = net_channel_growth
        # example: 
        # [net_input_channels=1, network_depth=2,
        # initial_channel_growth=32, channel_growth=2, net_channel_growth=3]:
        # channels in each layer: 1 -> 32 -> 64 -> 32 -> 3
        #
        self.conv_size = conv_size
        # rename pool_size -> downsam_size?
        self.pool_size = pool_size
        # upsam_size and pool_size should always be the same or positions from 
        # down- and up-pathwill not correspond during skip-connections
        # self.upsam_size = self.pool_size
        self.input_output_skip=input_output_skip
        #
        self.batch_renorm_scheme = None # TODO: only needed in train. Make arg to training fxns instead?  # scheme for clipping
        self.last_layer_batch_norm = self._set_last_layer_batch_norm(last_layer_batch_norm)
        #
        self.data_format = data_format
        # channel_axis is 1 or -1 depending on 'data_format'
        
        # TODO: am not sure about this, because that forbids it to later
        # provide conv_size as scalar such as conv_size=3
        ndims_spatial = len(conv_size)  # spatial dims
        # naming im_axis for semantic purposes.   Don't change value!
        self.im_axis = 0
        self.channel_axis = self._set_channel_axis()
        # TODO restructure this
        # TODO: remove mode -> from get_spatial_axes
        self.spatial_axes = utils.get_spatial_axes(
                "batch", ndims_spatial, im_axis=self.im_axis, 
                channel_axis=self.channel_axis)
        
        ## The following defines the network architecture
        # block order -> should not be changed.
        # TODO: rename network_depth -> n_downsampling
        self.block_list = (["input_block",] +  
                           (self.network_depth-1)*["down_block",] + 
                           ["bottom_block",] +  # this counts to network depth
                           (self.network_depth-1)*["up_block",] + 
                           ["output_block",])
        assert (len(self.block_list) - 1) / 2 == self.network_depth
        # block definitions -> blocks consist of layers
        # in current implementation each block should only have one conv_layer
        # or conv_block (added later: WHY?)
        self.block_definitions = dict()
        self.block_definitions["input_block"] = ["conv",] # no batch_norm  # TODO: style: change to input_conv and use definition in code other than shape determination
        self.block_definitions["down_block"] = ["conv_block", "skip_save", "pool"] #channel increase
        self.block_definitions["bottom_block"] = ["conv_block",] # no channel_increase
        self.block_definitions["up_block"] = ["upsam", "skip_concat", "conv_block"] #channel decrease
        self.block_definitions["output_block"] = ["conv",] # no activation  # TODO: style: change to output_conv and use definition in code other than shape determination
        # layer definitions -> layers consist of operations
        self.layer_definitions = dict()
        self.layer_definitions["conv_block"] = ["conv", "conv"]
        self.layer_definitions["skip_save"] = []
        self.layer_definitions["skip_concat"] = ["concat",]
        
        self.ready = False  # you need to set init or load from ckpt

    def extra_training_parameters(
            self, weight_init=None, batch_renorm_fn=None, dropout_rate=None):
        """
        Define how to initialize weights and batch_renorm (if used) in training
        mode. Not needed when loading from ckpt
        """
        self._set_weight_init(weight_init)
        self._set_batch_renorm_fn(batch_renorm_fn)
        self._set_dropout_rate(dropout_rate)
        self.set_ready()

    # %% build model from layers
    def inference(self, x_batch, is_training, print_info=True):
        """print info and build the network"""
        if not self.ready:
            warn("Model not set ready.  You need to run " +
                 "'unet3d.set_initialization(...)', when training new model. " + 
                 "To avoid this warning when loading from ckpt, run " +
                 "'Unet_v3.set_ready()'.")
        if is_training:
            print("building Unet_v3 for training")  # -> arch defines parameters
        else:
            print("building Unet_v3 for testing")
        # these are the parameters defined by arch
#        print("[n_channels_start:", self.n_channels_start, 
#              ", network depth:", self.network_depth, 
#              ", channel_growth:", self.channel_growth,
#              ", conv_size:", self.conv_size, 
#              ", pool_size:", self.pool_size, 
#              ", upsam_size:", self.upsam_size, 
#              ", padding:", self.padding, 
#              ", nonlinearity:" self.nonlinearity.__name__, " ,]")
#        if self.input_output_skip:
#            print("adding skip-connection between input and output.")

        net_input_shape = x_batch.shape
        if print_info:
            print("net input shape", net_input_shape)
        #self.check_net_input_shape(net_input_shape, mode="batch")  # TODO
        # model is built from self.block_list
        out = self._build_network(x_batch, is_training, print_info)
        if print_info:
            print("net output shape", out.shape)
        
        return out
    
    def _build_network(self, inputs, is_training, print_info=True):
        """
        self defines model,
        inputs and net_output_channels define input and output shapes,
        is_training defines dropout and batch_norm mode.
        -> NN defined
        """
        # store activations for skip-connections
        # output shapes from every block are needed for valid padding,
        # because activations from down-path must be cropped
        skip_connections = []
        skip_shapes = self._get_skip_shapes(inputs.shape)
        
        out = inputs
        for block_name in self.block_list:
            out, scope = self._build_block(out, block_name, is_training, 
                    skip_connections=skip_connections, skip_shapes=skip_shapes)
            if print_info:
                print(scope, out.shape)
        return out

#    def _get_scope(self, inputs, block_name):
#        in_num = inputs.shape[self.channel_axis]
#        out_num = self._get_out_num(in_num, block_name)
#        return block_name + str(in_num) + "_" + str(out_num)   
    
    # TODO: make block objects with methods:
    #    -> get_out_num
    #    -> get_scope
    
    def _build_block(self, inputs, block_name, is_training, 
                     skip_connections, skip_shapes):
        # determine out_num and scope
        in_num = utils.convert_shape_to_np_array(inputs.shape)[self.channel_axis]
        out_num = self._get_out_num(in_num, block_name)
        scope = block_name + str(in_num) + "_" + str(out_num)
        
        # input and output_block get extra treatment
        # TODO: I don't thinks this is necessary any more
        if block_name == "input_block":
            if self.input_output_skip:
                skip_connections.append(utils.crop(inputs, skip_shapes.pop()))
            out = self._input_layer(inputs, scope, is_training)
            # skip_connections.append(out) # other possibility without adding
            # but then I also need to take that factor into account in 
            # get_out_num !!
        elif block_name == "output_block":
            # out = self._concat(inputs, skip_connections.pop(), scope)
            out = self._output_layer(inputs, scope, is_training)
            if self.input_output_skip:
                out += skip_connections.pop()

        # U-Net body: down_block, bottom block, up_block
        else:
            out = inputs
            for layer in self.block_definitions[block_name]:
                if layer == "conv":
                    out = self._conv_layer(out, out_num, scope, is_training)
                elif layer == "conv_block":
                    out = self._conv_block(out, out_num, scope, is_training)
                elif layer == "skip_save":
                    skip_connections.append(utils.crop(out, skip_shapes.pop()))
                elif layer == "pool":
                    out = self._pool_layer(out, scope)
                elif layer == "upsam":
                    out = self._upsam_layer(out, scope)
                elif layer == "skip_concat":
                    out = self._concat(out, skip_connections.pop(), scope)
                else:
                    raise RuntimeError("Error building model. Unknown layer '" +
                                       layer + "'.")
        return out, scope

    # %% layers for building models
    
    # TODO: define in init which layers from op to use
    
    def _input_layer(self, inputs, scope, is_training):
        in_num = inputs.shape[self.channel_axis].value
        out_num = self._get_out_num(in_num, "input_block")   
        # TODO remove _get_out_num function or make it use the actual _input_layer
        return ops.input_layer_v3(
            inputs=inputs, 
            out_num=out_num, 
            conv_size=self.conv_size, 
            padding=self.padding, 
            is_training=is_training, 
            scope=scope+'/input_conv', 
            activation=self.nonlinearity, 
            init=self.weight_init, 
            data_format=self.data_format)

    def _output_layer(self, inputs, scope, is_training):
        in_num = inputs.shape[self.channel_axis].value
        out_num = self._get_out_num(in_num, "output_block")
        return ops.output_layer_v3(
            inputs=inputs, 
            out_num=out_num, 
            conv_size=self.conv_size, 
            padding=self.padding, 
            is_training=is_training, 
            scope=scope+'/output_conv',
            use_batch_renorm=self.use_batch_renorm and self.last_layer_batch_norm,
            use_batch_norm=self.use_batch_norm and self.last_layer_batch_norm,
            init=self.weight_init, 
            batch_renorm_scheme=self.batch_renorm_scheme,
            data_format=self.data_format)

    # TODO: try resnet block
    def _conv_block(self, inputs, out_num, scope, is_training):
        """channel growth is for entire block"""
#        in_num = inputs.shape[self.channel_axis].value
#        out_num = self._get_out_num(in_num, "conv_block")       
        activ1 = self._conv_layer(inputs, out_num, scope+'/conv1', is_training)
        return self._conv_layer(activ1, out_num, scope+'/conv2', is_training)

    def _conv_layer(self, inputs, out_num, scope, is_training):
#        self.layer_list.append("conv")
        return ops.conv_layer_v3(
            inputs=inputs, 
            out_num=out_num,
            conv_size=self.conv_size, 
            padding=self.padding, 
            is_training=is_training, 
            scope=scope+'/conv',
            activation=self.nonlinearity,
            use_dropout=self.use_dropout,
            dropout_rate=self.dropout_rate,
            use_batch_renorm=self.use_batch_renorm, 
            use_batch_norm=self.use_batch_norm,
            init=self.weight_init, 
            batch_renorm_scheme=self.batch_renorm_scheme,
            data_format=self.data_format)

    # TODO: try strided conv
    def _pool_layer(self, inputs, scope):
#        self.layer_list.append("pool")
        return ops.avg_pool(  #try avg pool
            inputs=inputs,
            pool_size=self.pool_size,
            padding=self.padding,
            scope=scope+'/pool',
            data_format=self.data_format)
        
    def _upsam_layer(self, inputs, scope):
#        self.layer_list.append("upsam")
        return ops.upsample_NN(  # try FT upsam; try extra conv after upsam and before concat
            inputs=inputs,
            upsam_size=self.pool_size,
            scope=scope+'/upsam', 
            data_format=self.data_format)

    def _concat(self, in1, in2, scope):
#        self.layer_list.append("concat")
        return tf.concat([in1, in2], axis=self.channel_axis, name=scope+'/concat')

    # %% internal setters
    def set_ready(self):
        self.ready=True
        return self.ready

    def _set_dropout_rate(self, dropout_rate):
        self.dropout_rate = dropout_rate
#        if self.dropout_rate == 0.0:
#            self.use_dropout = False
        return self.dropout_rate
        
    def _set_channel_axis(self):
        if self.data_format in ["channels_last", "NDHWC"]:
            # + 2 for im_axis, ch_axis  and  
            # -1 because of zero-indexing in python -> + 1
            self.channel_axis = -1 #len(self.spatial_axes) + 1
        elif self.data_format in ["channels_first", "NCDHW"]:
            self.channel_axis = 1
        return self.channel_axis

    def _set_batch_renorm_fn(self, batch_renorm_fn):
        self.batch_renorm_fn = batch_renorm_fn
        return self.batch_renorm_fn
    
    def _set_weight_init(self, weight_init):
        self.weight_init = weight_init
        return self.weight_init
    
    def _set_last_layer_batch_norm(self, last_layer_batch_norm):
        self.last_layer_batch_norm = last_layer_batch_norm 
        if self.last_layer_batch_norm is None:  # default behavior
            if self.use_batch_norm or self.use_batch_renorm:
                self.last_layer_batch_norm = True
            else:
                self.last_layer_batch_norm = False
        if (self.last_layer_batch_norm and 
            not(self.use_batch_renorm or self.use_batch_norm)):
            warn("last_layer_batch_norm only has an effect, " +
                 "when use_batch_renorm or use_batch_norm is True.")
        return self.last_layer_batch_norm
    
    # %% wrappers for utils

    def pretend_network_pass(
            self, t, override_padding=None, exclude_channel_axis=True):
        """
        pretend t was propagated through network.  
        Will change shape of t, if necessary.
        You can provide override_padding to pretend using a net with different 
        padding.
        Excludes channels by default meaning that this will not change the 
        number of channels according to net_channel_growth
        This is needed eg for cut_loss_to_valid
        """
        if override_padding:  # needed for cut_loss_to_valid
            #TODO this is a bit ugly.  
            # rather allow override_padding args to preprocesss_shape and _crop_to_valid
            orig_padding = self.padding
            self.padding = override_padding
        with tf.name_scope("pretend_network_pass"):
            if self.padding == "valid":
                t = self.preprocess_input_shape(t)  # just spatial
                t = self._crop_to_valid_net_output_shape(t)                    
        if override_padding:
            self.padding = orig_padding
        # Do nothing for same padding.
        return t

    def _crop_to_valid_net_output_shape(self, t, exclude_channel_axis=True):
        """
        This removes all pixels that were affected by padding from output.
        It is meant to be used for inputting y into the loss function for valid
        padding or for same padding with cut_loss_to_valid.
        """
        out_shape = self._get_net_output_shape(t.shape)
        if exclude_channel_axis:
            out_shape[self.channel_axis] = None
        return utils.crop(t, out_shape)
    
    
    # TODO !!! Debug this
    def preprocess_input_shape(self, input_image):
        # input image must be in batch format
        
        # check if input image has been batched
        in_shape = utils.convert_shape_to_np_array(input_image.shape)
        if len(in_shape) == len(self.spatial_axes) + 1:
            # assume input_image has not been batched (dhwc) -> add n
            in_shape = np.array([None] + list(in_shape))
        elif len(in_shape) == len(self.spatial_axes) + 2:
            # assume input_image has been batched (ndhwc)
            # nothing to be done
            # print(in_shape)  # possibly set im_axis to 0
            pass
        else: 
            # TODO improve error message
            raise ValueError(
                    "input_image does not have the right shape.")
            
        nearest_smaller_shape, nearest_larger_shape = \
                self._get_nearest_allowed_shapes(in_shape)        
        if np.all(nearest_smaller_shape == nearest_larger_shape):
            return input_image
        elif self.padding == "same":
            # TODO: move print out of here so that it is not called during pretend_network_pass
            # or make print more informative
            print("padding with zeroes on the right to nearest allowed " +
                  "input image size.")
            input_image = utils.pad_right(input_image, nearest_larger_shape)
            #self._pad_to_nearest_allowed_input_shape(input_image)
        elif self.padding == "valid":
            # TODO: move print out of here so that it is not called during pretend_network_pass
            # or make print more informative
            print("Cropping to nearest allowed input image size.")
            input_image = utils.crop(input_image, nearest_smaller_shape)
            #self._crop_to_nearest_allowed_input_shape(input_image)  # TODO
            
        return input_image

    # TODO: move body to utils
    def _get_nearest_allowed_shapes(self, in_shape, max_pixels=None):
        """returns pair (nearest_smaller_shape, nearest_larger_shape)"""
        # assumes batch input
        
        in_shape = utils.convert_shape_to_np_array(in_shape)
        ndims = len(in_shape) - 2
        spatial_axes = self.spatial_axes

        batch_size = in_shape[self.im_axis]
        net_input_channels = in_shape[self.channel_axis]
        
        # create minimal bottom block output shape
        if self.padding == "same":        
            # bottom layer output can be any number >= 1
            bottom_block_output_pixels = 1
        elif self.padding == "valid":
            # bottom_layer_output must be:
            # -> even st. output from all down layers is even
            # -> and > 4 st. network can expand in up-path 
            #   (TODO: exact value depends on architectures though).
            #    this is just guaranteed to work for 3x3x3 convs with 2x2x2 pooling
            bottom_block_output_pixels = 6
        bottom_block_output_channels = (
                net_input_channels * self.initial_channel_growth * 
                self.channel_growth**(self.network_depth-1))
        bottom_block_output_shape = (
                [batch_size] + ndims*[bottom_block_output_pixels])
        # insert like this because channel axis may be -1 or 1
        # TODO: ensure ch_ax is not <1 during  init
        # or at the beginning of inference
        ch_axis = self.channel_axis
        if ch_axis < 0:
            ch_axis = len(bottom_block_output_shape) + self.channel_axis + 1
        bottom_block_output_shape.insert(ch_axis, bottom_block_output_channels)
        bottom_block_output_shape = np.array(bottom_block_output_shape)
        
        if max_pixels is None:
            max_pixels = np.inf
        
        # get the closest larger and smaller allowed shape to in_shape.
        # previous and next allowed size will be equal in case in_shape is allowed
        
        # init prev with minimal shape
        # TODO: prev is actually not necessary for same.
        # consider returning None
        prev_allowed_sh = self._get_net_input_shape_from_bottom_block_output_shape(
                bottom_block_output_shape)        
        if np.any(in_shape[spatial_axes] < prev_allowed_sh[spatial_axes]):
            raise ValueError(
                    "The input shape " + str(in_shape) + " is smaller than the " +
                    "minimum allowed net input shape" + str(prev_allowed_sh) + 
                    "along at least one spatial dimension.")       
        
        # init next with prev
        next_allowed_sh = prev_allowed_sh
        if np.any(next_allowed_sh[spatial_axes] > max_pixels):
            raise ValueError(
                    "The minimum allowed input size " + str(next_allowed_sh) + 
                    "is smaller than the largest considered number of pixels " +
                    str(max_pixels) + "along at least one spatial dimension.") 

        while spatial_axes:
            prev_allowed_sh[spatial_axes] = next_allowed_sh[spatial_axes]
            if np.any(in_shape[spatial_axes] == prev_allowed_sh[spatial_axes]):
                mask = in_shape == next_allowed_sh
                spatial_axes = [i for i in spatial_axes if not mask[i]]
            
            if self.padding == "same":
                bottom_block_output_shape[spatial_axes] += 1
            elif self.padding == "valid":
                bottom_block_output_shape[spatial_axes] += 2
            else:
                raise ValueError("unknown padding.")
            next_allowed_sh = self._get_net_input_shape_from_bottom_block_output_shape(
                    bottom_block_output_shape)  
            if np.any(in_shape[spatial_axes] < next_allowed_sh[spatial_axes]):
                mask = in_shape < next_allowed_sh
                spatial_axes = [i for i in spatial_axes if not mask[i]]
                    
        return prev_allowed_sh, next_allowed_sh

    
    def _get_net_input_shape_from_bottom_block_output_shape(self, 
                bottom_block_out_shape):
        """
        Determine the input shape from the shape of the output of the bottom 
        layer. This function is used to calculate shapes that can be used as 
        input to unet.  
        Allowed input to unet must have even block_input_shape in every layer.
        """    
        used_correct_pool_inputs = True # since am starting from bottom block
        # sequence of entire down path including bottom_block
        sh = bottom_block_out_shape
        # block_list should contain: input_block, down_block, bottom_blocks,
        # up_blocks, output_block 
        n_down_blocks = self.network_depth + 1  # incl. input and bottom
        blocks = self.block_list[:n_down_blocks]
        blocks = blocks[::-1]  # starting from bottom block
        for block in blocks:
            bdef = self.block_definitions[block]
            in_num = self._get_in_num(sh[self.channel_axis], block)
            sh = self._get_layer_sequence_input_shape(
                    sh, bdef, in_num, 
                    used_correct_pool_inputs=used_correct_pool_inputs)
        return sh
    
    def _get_net_output_shape(self, in_shape):
        """get output shape after fwd-prop through unet"""
        return self._get_block_output_shapes(in_shape)[-1]
 
#    def _get_block_input_shapes(self, out_shape, net_input_channels):
#        """get a list of the shapes before each block in Unet_v3."""
#        sh = out_shape
#        block_input_shapes = list()
#        for block in self.block_list:
#            bdef = self.block_definitions[block]
#            in_num = self._determine_in_num(sh[self.channel_axis], block, 
#                                    net_input_channels=net_input_channels)
#            sh = self._get_layer_sequence_input_shape(sh, bdef, in_num)
#            block_input_shapes.append(sh)
#        return block_input_shapes

    def _get_block_output_shapes(self, in_shape):
        """get a list of the shapes after each block in Unet_v3."""
        sh = in_shape
        block_output_shapes = list()
        for block in self.block_list:
            bdef = self.block_definitions[block]
            out_num = self._get_out_num(sh[self.channel_axis], block)
            sh = self._get_layer_sequence_output_shape(sh, bdef, out_num)
            block_output_shapes.append(sh)
        return block_output_shapes

    def _get_skip_shapes(self, in_shape):
        """get a list of the shapes for concatenating skip_connetions in Unet_v3."""
        in_shape = utils.convert_shape_to_np_array(in_shape)
        sh = in_shape
#        ch_axis = self.channel_axis
#        if ch_axis < 0:
#            ch_axis = len(in_shape) + self.channel_axis
        
        skip_out_shapes = list()
#        skip_in_shapes = list()
        for block in self.block_list:
            bdef = self.block_definitions[block]
            out_num = self._get_out_num(sh[self.channel_axis], block)
            # get indices of skip_save (possibly more than one)
            skip_concat_idx = [i for i, layer in enumerate(bdef) if layer=='skip_concat']
#            skip_save_idx = [i for i, layer in enumerate(bdef) if layer=='skip_save']
            if not skip_concat_idx:# and not skip_save_idx:
                # just determine block out shape
                sh = self._get_layer_sequence_output_shape(sh, bdef, out_num)
#            elif skip_save_idx:
#                # sequence until skip concat
#                seq = bdef[:skip_save_idx[0]]     
#                sh = self._get_layer_sequence_output_shape(sh, seq, out_num)
#                skip_in_shapes.append(sh)
#                # will only enter for-loop if len(skips)>1
#                for i in range(len(skip_save_idx)-1):
#                    seq = bdef[skip_save_idx[i]:skip_save_idx[i+1]]
#                    sh = self._get_layer_sequence_output_shape(sh, seq, out_num)
#                    skip_in_shapes.append(sh)
#                # sequence after skip concat
#                seq = bdef[skip_save_idx[-1]:]     
#                sh = self._get_layer_sequence_output_shape(sh, seq, out_num)
            elif skip_concat_idx:
                # sequence until skip concat
                seq = bdef[:skip_concat_idx[0]]     
                sh = self._get_layer_sequence_output_shape(sh, seq, out_num)
                skip_out_shapes.append(sh)
                # will only enter for-loop if len(skips)>1
                for i in range(len(skip_concat_idx)-1):
                    seq = bdef[skip_concat_idx[i]:skip_concat_idx[i+1]]
                    sh = self._get_layer_sequence_output_shape(sh, seq, out_num)
                    skip_out_shapes.append(sh)
                # sequence after skip concat
                seq = bdef[skip_concat_idx[-1]:]     
                sh = self._get_layer_sequence_output_shape(sh, seq, out_num)
#        if len(skip_out_shapes) != len(skip_in_shapes):
#            raise RuntimeError("There are " + str(len(skip_out_shapes)) + 
#                               "skip_concats and " + str(len(skip_in_shapes)) +
#                               "skip_saves.  This needs to match.")
#        skip_in_shapes = skip_in_shapes[::-1]
#        skip_shapes = list()
#        for i in range(len(skip_out_shapes)):
#            skip = skip_out_shapes[i]
#            skip[self.im_axis] = skip_in_shapes[i][self.im_axis]
#            skip[self.channel_axis] = skip_in_shapes[i][self.channel_axis]
#            skip_shapes.append(skip)
#        list([skip_in_shapes[self.im_axis]] + 
#                           [skip_out_shapes[spatial_axes])
#        skip_shapes.insert(ch_axis, skip_in_shapes[self.im_axis])
        if self.input_output_skip:
            skip_out_shapes.append(self._get_net_output_shape(in_shape))
        return skip_out_shapes
    
    def _get_out_num(self, in_num, block):
        if block == "input_block":
            out_num = in_num*self.initial_channel_growth
        elif block == "down_block":
            out_num = in_num * self.channel_growth
        elif block == "bottom_block":
            out_num = in_num
        elif block == "up_block":
            # assuming this is to undo increase during down-path, this should
            # never give error
            out_num, rem = divmod(in_num, self.channel_growth)
            if rem != 0:
                raise ValueError(
                        "Error in " + block + ". number of input channels " +
                        str(in_num) + "is not divisible by channel_growth" + 
                        str(self.channel_growth) + ".")
        elif block == "output_block":
            out_num, rem = divmod(in_num, self.initial_channel_growth)
            if rem != 0:
                raise ValueError(
                        "Error in " + block + ". number of input channels " +
                        str(in_num) + " is not divisible by initial_channel_growth " +
                        str(self.initial_channel_growth) + ". Did you use the " +
                        "output_block before the input_block?")
            out_num *= self.net_channel_growth
        else:
            raise ValueError("Unknown block '" + block + "' given.")
        return out_num

    def _get_in_num(self, out_num, block):
        if block == "input_block":
            in_num, rem = divmod(out_num, self.initial_channel_growth)
            if rem != 0:
                raise ValueError(
                        "Error in ", block, ". number of output channels " +
                        str(out_num) + " is not divisible by initial_channel_growth", 
                        str(self.initial_channel_growth) + ".")
        elif block == "down_block":
            in_num, rem = divmod(out_num, self.channel_growth)
            if rem != 0:
                raise ValueError(
                        "Error in " + block + ". number of output channels" +
                        out_num + "is not divisible by channel_growth" + 
                        self.channel_growth + ".")
        elif block == "bottom_block":
            in_num = out_num
        elif block == "up_block":
            in_num = out_num * self.channel_growth
        elif block == "output_block":
            in_num, rem = divmod(in_num, self.net_channel_growth)
            if rem != 0:
                raise ValueError(
                        "Error in ", block, ". number of output channels",
                        out_num, "is not divisible by net_channel_growth", 
                        self.net_channel_growth + ".")
            in_num *= self.initial_channel_growth
        else:
            raise ValueError("Unknown block '" + block + "' given.")
        return in_num
    
    def _get_layer_sequence_output_shape(self, in_shape, sequence, out_num):
        in_shape = utils.convert_shape_to_np_array(in_shape)
        # split compound-layers in sequence
        layers = list()
        for layer in sequence:
            if layer in ["conv_block", "skip_save", "skip_concat"]:
                layer = self.layer_definitions[layer]
                layers += layer
            else:
                layers.append(layer)
        out_shape = list(utils.get_layer_sequence_output_shape(
                in_shape, layers, out_num, padding=self.padding, 
                conv_size=self.conv_size, pool_size=self.pool_size, 
                upsam_size=self.pool_size, im_axis=self.im_axis,
                channel_axis=self.channel_axis))
        return out_shape

    def _get_layer_sequence_input_shape(self, out_shape, sequence, in_num, 
                                        used_correct_pool_inputs=False):
        layers = list()
        for layer in sequence:
            if layer in ["conv_block", "skip_save", "skip_concat"]:
                layer = self.layer_definitions[layer]
                layers += layer
            else:
                layers.append(layer)
        return utils.get_layer_sequence_input_shape(
                out_shape, layers, in_num, padding=self.padding, 
                conv_size=self.conv_size, pool_size=self.pool_size, 
                upsam_size=self.pool_size, 
                used_correct_pool_inputs=used_correct_pool_inputs,
                im_axis=self.im_axis, channel_axis=self.channel_axis)
