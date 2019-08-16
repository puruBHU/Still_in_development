#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 22:47:35 2019

@author: Purnendu Mishra

This test areas will verify that whether following functions functionality is same as that 
present in box_utils.py
    jaccard
    encode
    match
    target output from numpy function should be same as that of torch function
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

from CustomDataLoaderTF import DataAugmentor as DA2
from CustomDataLoaderTorch import DataAugmentor as DA1


from utility import match as match_np
from box_utils import match as match_th

from pathlib import Path
import collections
from ssd300_model import SSD300
from skimage.io import imread, imshow
import cv2
import numpy as np

#root                = Path.home()/'data'/'VOCdevkit'/'VOC2007'
root = Path.home()/'Documents'/'DATASETS'/'VOCdevkit'

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

priors = generate_ssd_priors(specs).astype(np.float32)


batch_size = 16

loader_th  = DA1(rescale=1/255.0)
data_th    = loader_th.flow_from_directory(root         = root,
                                       data_folder  = ['VOC2007', 'VOC2012'],
                                       target_size  = (300,300),
                                       batch_size   = batch_size,
                                       shuffle      = False,
                                       priors       = priors
                                        )


loader_np  = DA2(rescale=1/255.0)
data_np    = loader_np.flow_from_directory(root         = root,
                                       data_folder  = ['VOC2007', 'VOC2012'],
                                       target_size  = (300,300),
                                       batch_size   = batch_size,
                                       shuffle      = False,
                                       priors       = priors
                                        )

priors_th  = data_th.priors
priros_np  = data_np.priors

is_equal = priros_np == priors_th.numpy()
print(np.sum(is_equal))

sample_th = data_th[0]
images_th, targets_th = sample_th

targets_th = np.array(targets_th)

sample_np = data_th[0]
images_np, targets_np = sample_np
targets_np = np.array(targets_np)

is_equal_target = targets_np == targets_th
print(np.sum(is_equal_target))