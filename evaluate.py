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
def preprocess_image(image_path, target_size=(300, 300), mean = None, std = None):
    image = imread(image_path)
    img = cv2.resize(image, (300, 300), interpolation = cv2.INTER_CUBIC)
    img = np.expand_dims(img,axis = 0)
    img = np.float32(img)
    img -= mean
    img /= std
    return image, img

#%%****************************************************************************
model = SSD300(input_shape = (300, 300, 3), 
               anchors = [4, 6,6,6,4,4], 
               num_classes = 21)

model.load_weights('checkpoints/test_checkpoint.hdf5')

#%%****************************************************************************

image_path  =  Path.cwd()/'test_images'/'000007.jpg'

image, input_img = preprocess_image(image_path = image_path,
                                    mean = mean, 
                                    std = std)

#%%****************************************************************************
prediction  = model.predict(input_img) 
prediction  = np.squeeze(prediction, axis=0)

loc_pred  = prediction[:,:4]
conf_pred = prediction[:,4:]