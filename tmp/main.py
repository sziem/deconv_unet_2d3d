import os
import time
import argparse
import tensorflow as tf
#from network import Unet_3D

"""
This file provides configuration to build U-NET for semantic segmentation.
"""

def configure():
    flags = tf.app.flags
    
    # configuration
    # not used // does not work
    # flags.DEFINE_string('DatasetHandler', 'DatasetHandler', 'which dataset_handler to use')
    
    # training
    flags.DEFINE_string('training_data_path', './testdata/vascu_pairs_train.h5', 'path to training data')
    flags.DEFINE_integer('max_step', 250000, '# of step for training')  # TODO: use epochs instead1
    flags.DEFINE_integer('save_interval', 1000, '# of interval to save model')
    flags.DEFINE_integer('summary_interval', 100, '# of step to save summary')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    
    # data
    #flags.DEFINE_string('data_dir', os.path.join("testdata","dataset.h5"), 'Name of data file(s)')
    flags.DEFINE_integer('batch_size', 10, 'training batch size')
   
    # logging
    flags.DEFINE_string('logdir', './logs', 'Saving logs to this dir')
    flags.DEFINE_string('modeldir', './ckpts', 'Saving ckpts to this dir')
    flags.DEFINE_string('savedir', './results', 'Saving deconvolved images to this dir')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    #flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    
    # network architecture
#    flags.DEFINE_integer('network_depth', 4, 'network depth for U-Net')
#    flags.DEFINE_integer('start_channel_num', 32,
#                         'start number of outputs for the first conv layer')
    flags.DEFINE_string('action', 'concat',
        'Use how to combine feature maps in pixel_dcl and ipixel_dcl: concat or add')
   
    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', dest='option', type=str, default='train',
                        help='actions: train, test, or predict')
    args = parser.parse_args()
    if args.option not in ['train', 'test', 'predict']:
        print('invalid option: ', args.option)
        print("Please input a option: train, test, or predict")
    else:
        model = Unet_3D(tf.Session(), configure())
        getattr(model, args.option)()  #model.train() or test() or predict()


if __name__ == '__main__':
    # configure which gpu or cpu to use
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    tf.app.run()  # you may add argvs here!
