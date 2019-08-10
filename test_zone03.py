#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:39:32 2019

@author: Purnendu Mishra

"""
import torch
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

from CustomDataLoaderv3 import DataAugmentor as DA1
from CustomDataLoader import DataAugmentor as DA2

from utility import match as match_np
from utility import decode as decode_np
from utility import non_maximum_supression

from box_utils import match as match_th
from box_utils import decode as decode_th
from box_utils import nms

from pathlib import Path
import collections
from ssd300_model import SSD300
from skimage.io import imread, imshow
import cv2
import numpy as np

#root                = Path.home()/'data'/'VOCdevkit'/'VOC2007'
root                 = Path.home()/'Documents'/'DATASETS'/'VOCdevkit'/'VOC2007'
voc_2007_datafile  = root/'ImageSets'/'Main'/'train.txt'

voc_2007_images      = root/'JPEGImages'
voc_2007_annotations = root/'Annotations'


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


batch_size = 4

loader_th  = DA1()
data_th    = loader_th.flow_from_directory(root     = root,
                                        data_file   = voc_2007_datafile,
                                        target_size = 300,
                                        batch_size  = batch_size,
                                        shuffle     = False,
                                        num_classes = 21,
                                        priors      = priors
                                        )


loader_np  = DA2()
data_np    = loader_np.flow_from_directory(root     = root,
                                        data_file   = voc_2007_datafile,
                                        target_size = 300,
                                        batch_size  = batch_size,
                                        shuffle     = False,
                                        num_classes = 21,
                                        priors      = priors
                                        )


sample_np = data_th[0]
images_np, targets_np = sample_np

loc_data  = targets_np[:,:,:4]
conf_data = targets_np[:,:,4:]

a = loc_data[0,:,:]

decoded_np = decode_np(loc = a, priors=priors, variances=[0.1, 0.2]) 

a_      = torch.from_numpy(a).float()
priors_ = torch.from_numpy(priors).float()

decoded_th = decode_th(loc = a_, priors=priors_, variances=[0.1, 0.2])
c = decoded_th.numpy() == decoded_np
print(np.sum(c))


scores = np.random.rand(8732,)

scores_ = torch.from_numpy(scores).float()

nms_np = non_maximum_supression(boxes = decoded_np, scores= scores, top_k=200, overlap=0.5)
nms_th = nms(boxes=decoded_th, scores= scores_, overlap=0.5, top_k=200)

print(nms_th[0])
print(nms_np[0])