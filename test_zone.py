#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:58:26 2019

@author: puru
"""

from keras import backend as K

from SSD_generate_anchors import generate_ssd_priors
from CustomDataLoader import DataAugmentor
from utility import *
from pathlib import Path
import collections


#root = Path.home()/'data'/'VOCdevkit'/'VOC2007'
root  = Path.home()/'Documents'/'DATASETS'/'VOCdevkit'/'VOC2007'
voc_2007_datafile = root/'ImageSets'/'Main'/'train.txt'
voc_2007_images   = root/'JPEGImages'
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

priors = generate_ssd_priors(specs)
testloader = DataAugmentor()

data       = testloader.flow_from_directory(root = root,
                                            data_file=voc_2007_datafile,
                                            target_size=300,
                                            batch_size = 4,
                                            shuffle    = True
                                            )

sample = data[0]

image, target = sample
batch_size = image.shape[0]

p = point_form(priors)

for i in range(batch_size):
    t = point_form(target[i][:,1:])
    label = target[i][:,0]
    
    iou = jaccard(t, p)

best_prior_overlap = np.amax(iou, axis=-1).astype(np.float32)
best_prior_idx     = np.argmax(iou, axis =-1)

best_truth_overlap = np.amax(iou, axis=0).astype(np.float32)
best_truth_idx     = np.argmax(iou, axis = 0)

for j in range(best_prior_idx.shape[0]):
    best_truth_idx[best_prior_idx[j]] = j
    
matches = t[best_truth_idx]
conf    = label[best_truth_idx]
