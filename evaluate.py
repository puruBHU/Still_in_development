#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:48:45 2019

@author: Purnendu Mishra
"""



import cv2
from pathlib import Path

from ssd300_model import SSD300
from skimage.io import imread
import numpy as np
from utility import *

import collections
from nms import nms

from SSD_generate_anchors import generate_ssd_priors
import matplotlib.pyplot as plt
#%%****************************************************************************
target_size = (300,300)

mean = np.array([114.02898, 107.86698,  99.73119], dtype=np.float32)
std  = np.array( [69.89365, 69.07726, 72.30074], dtype=np.float32)

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

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
def preprocess_image(image_path, target_size=(300, 300), mean = None, std = None):
    image = imread(image_path)
    img = cv2.resize(image, (300, 300), interpolation = cv2.INTER_CUBIC)
    img = np.expand_dims(img,axis = 0)
    img = np.float32(img)
    img -= mean
    img /= std
    return image, img

def TransformCoordinates(boxes = None, image_size = (300, 300)):
    xmin = np.maximum(0, boxes[:,0]) * image_size[1]
    ymin = np.maximum(0, boxes[:,1]) * image_size[0]
    
    xmax = np.maximum(0, boxes[:,2]) * image_size[1]
    ymax = np.maximum(0, boxes[:,3]) * image_size[0]
    
    return (xmin.astype(np.int16), ymin.astype(np.int16), xmax.astype(np.int16), ymax.astype(np.int16))

#%%****************************************************************************
model = SSD300(input_shape = (300, 300, 3), 
               anchors = [4, 6,6,6,4,4], 
               num_classes = 21)

model.load_weights('checkpoints/test_checkpoint.hdf5')

#%%****************************************************************************

image_path  =  Path.cwd()/'test_images'/'000012.jpg'

image, input_img = preprocess_image(image_path = image_path,
                                    mean = mean, 
                                    std = std)

#%%****************************************************************************
prediction  = model.predict(input_img) 
prediction  = np.squeeze(prediction, axis=0)

loc_pred  = prediction[:,:4]
conf_pred = prediction[:,4:]

predicted_class_id = np.argmax(conf_pred, axis = -1)
predicted_classes = np.unique(predicted_class_id)

loc_pred_ = decode(loc =loc_pred , priors=priors, variances= [0.1,0.2])

selected_boxes = nms(boxes = loc_pred_, overlapThresh=0.5)

xmin, ymin, xmax, ymax = TransformCoordinates(boxes = selected_boxes)

for i in range(selected_boxes.shape[0]):
    cv2.rectangle(image, (xmin[i], ymin[i]), (xmax[i], ymax[i]), (255, 0, 0), 2)

plt.imshow(image)
plt.show()


