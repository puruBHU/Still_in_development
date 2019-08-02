#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:04:54 2019

@author: Purnendu Mishra
"""

import tensorflow as tf


from keras.backend.tensorflow_backend import set_session
##********************************************************
## For GPU
#
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.50
set_session(tf.Session(config=config))

#
##********************************************************

from keras import backend as K

from SSD_generate_anchors import generate_ssd_priors
from CustomDataLoader import DataAugmentor
from pathlib import Path

import collections
from ssd300_model import SSD300

from ssd_loss_function_v2 import CustomLoss
from keras.optimizers import SGD
import numpy as np

tf_session = K.get_session()


#%%****************************************************************************
root                  = Path.home()/'data'/'VOCdevkit'/'VOC2007'
#root                  = Path.home()/'Documents'/'DATASETS'/'VOCdevkit'/'VOC2007'
voc_2007_train_file   = root/'ImageSets'/'Main'/'train.txt'
voc_2007_val_file     = root/'ImageSets'/'Main'/'val.txt'
voc_2007_images       = root/'JPEGImages'
voc_2007_annotations  = root/'Annotations'




#%%****************************************************************************
mean = np.array([114.02898, 107.86698,  99.73119], dtype=np.float32)
std  = np.array( [69.89365, 69.07726, 72.30074], dtype=np.float32)

target_size = (300,300)
batch_size  = 4

num_epochs  = 5

#%%****************************************************************************
SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])

Spec = collections.namedtuple('Spec', ['feature_map_size', 'shrinkage', 'box_sizes', 
                                       'aspect_ratios'])

# the SSD orignal specs
specs = [
    Spec(38, 8, SSDBoxSizes(30, 60), [2]),
    Spec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
    Spec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
    Spec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
    Spec(3, 100, SSDBoxSizes(213, 264), [2]),
    Spec(1, 300, SSDBoxSizes(264, 315), [2])
]

priors = generate_ssd_priors(specs)

#%%****************************************************************************
trainloader = DataAugmentor(rescale    = 1.,
                            normalize  = True,
                            mean       = mean,
                            std        = std)

valloader  = DataAugmentor(rescale    = 1.,
                            normalize  = True,
                            mean       = mean,
                            std        = std)


train_generator   = trainloader.flow_from_directory(
                                            root        = root,
                                            data_file   = voc_2007_train_file,
                                            target_size = target_size,
                                            batch_size  = batch_size,
                                            shuffle     = True
                                            )

val_generator     = valloader.flow_from_directory (
                                            root         = root,
                                            data_file    = voc_2007_train_file,
                                            target_size  = target_size,
                                            batch_size   = batch_size,
                                            shuffle      = False
                                            )


steps_per_epoch  = len(train_generator)
validation_steps = len(val_generator)
#%*****************************************************************************
model = SSD300(input_shape=(300,300, 3), num_classes=21)
#model.summary()

#lossFN = SSDLoss(anchors   = priors,
#                 threshold = 0.6,
#                 neg_pos   = 3,
#                 num_classes = 21,
#                 alpha       = 1.0,
#                 variance    = [0.1,0.2])


model.compile(optimizer = SGD(lr= 1e-3, momentum = 0.9 , nesterov=True, decay=1e-5),
              loss      = CustomLoss(anchors     = priors,
                                     negpos_ratio= 3,
                                     num_classes = 21,
                                     alpha       = 1.0
                                     ))

#%%****************************************************************************
#model.fit_generator(generator       =  train_generator,
#                    validation_data =  val_generator,   
#                    epochs          =  num_epochs,
#                    steps_per_epoch = steps_per_epoch,
#                    validation_steps = validation_steps,
#                    verbose   = 1,
#                    shuffle   = True,
#                    use_multiprocessing  = True,
#                    workers              = 4)

