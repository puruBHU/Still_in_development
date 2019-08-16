#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 12:18:54 2019

@author: Purnendu Mishra
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from random import shuffle
from CustomDataLoaderTesting import DataAugmentor
from SSD_generate_anchors import generate_ssd_priors
from utility import point_form
import collections

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

priors = generate_ssd_priors(specs)

loader = DataAugmentor(horizontal_flip=True,
                       )

batch_size = 5
generator = loader.flow_from_directory(root         = root,
                                       data_folder  = ['VOC2007', 'VOC2012'],
                                       target_size  = (300,300),
                                       batch_size   = batch_size,
                                       shuffle      = True,
                                       priors       = priors)

image, target = generator[0]



#fig = plt.figure(figsize=(15,15))
#
#for i in range(batch_size):
#    image_  = image[i].astype(np.uint8)
#    h, w,c  = image_.shape
#    boxes   = target[i]
##    print(boxes)
#    boxes   = point_form(boxes[:,1:])
#    boxes = np.array(boxes)
#    
#    ax = fig.add_subplot(batch_size,1,i+1)
#   
#    for j in range(boxes.shape[0]):
#        box =  boxes[j,:]
#        xmin = int(box[0] * w)
#        ymin = int(box[1] * h)
#
#        xmax = int(box[2] * w)
#        ymax = int(box[3] * h)
#        
#  
#        cv2.rectangle(image_, (xmin,ymin ), (xmax ,ymax),  (255,0,0), 2)
#        
##    plt.subplot(1,batch_size,i+1)
#    ax.imshow(image_)
#      
#plt.show()
