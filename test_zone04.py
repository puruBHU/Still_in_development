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
from utility import point_form


root = Path.home()/'Documents'/'DATASETS'/'VOCdevkit'

loader = DataAugmentor(horizontal_flip=True,
                       vertical_flip  =False)

batch_size = 1
generator = loader.flow_from_directory(root         = root,
                                       data_folder  = ['VOC2007', 'VOC2012'],
                                       target_size  = (300,300),
                                       batch_size   = batch_size,
                                       shuffle      = True)

image, target = generator[0]

#def point_form(boxes):
#    
#    #xmin  = xc - w/2
#    #xmax = xmin + w
#
#    xc = boxes[0]
#    yc = boxes[1]
#    
#    h  = boxes[2]
#    w  = boxes[3]
#    
#    xmin = xc - w/2
#    xmax = xmin + w
#    
#    ymin = yc - h/2
#    ymax = ymin + h
#    
#    return xmin, ymin, xmax, ymax 


for i in range(batch_size):
    image = image[i].astype(np.uint8)
    h, w, c = image.shape 
    
    boxes = target[i]
    boxes   = point_form(boxes[:,1:])
    for j in range(boxes.shape[0]):
        box =  boxes[j,:]
        xmin = int(box[0] * w)
        ymin = int(box[1] * h)

        xmax = int(box[2] * w)
        ymax = int(box[3] * h)
        
        cv2.rectangle(image,(xmax ,ymax), (xmin,ymin ), (255,0,0), 2)
    
    plt.imshow(image)
    
plt.show()
