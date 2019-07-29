#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:10:11 2019

@author: Purnendu Mishra
"""
from keras.layers import Conv2D, SeparableConv2D, Activation
from keras.layers import BatchNormalization
from keras.initializers import he_normal
from keras.regularizers import l2
from keras import backend as K

import numpy as np

def _bn_relu(input_):
    norm = BatchNormalization()(input_)
    return Activation('relu')(norm)

def conv_bn_relu(**params):
    filters     = params['filters']
    kernel_size = params.setdefault('kernel_size', (3,3))
    strides     = params.setdefault('strides',(1,1))
    padding     = params.setdefault('padding','same')
    dilation_rate = params.setdefault('dilation_rate', 1)
    kernel_initializer = params.setdefault('kernel_initializer', he_normal())
    kernel_regularizer = params.setdefault('kernel_regularizer', l2(1e-3))
    activation         = params.setdefault('activation','relu')
    name               = params.setdefault('name', None)

    def f(input_):
        conv = Conv2D(filters       = filters,
                      kernel_size   = kernel_size,
                      strides       = strides,
                      padding       = padding,
                      dilation_rate = dilation_rate,
                      kernel_initializer = kernel_initializer,
                      kernel_regularizer = kernel_regularizer,
                      name = name['conv'])(input_)

        batch_norm = BatchNormalization(name = name['batch_norm'])(conv)

        return Activation(activation,name = name['activation'])(batch_norm)
    return f


def point_form(boxes):
    """ Convert prior boxes to (xmin, ymin, xmax, ymax)
    """
    xc  = boxes[0]
    yc  = boxes[1]
    
    w   = boxes[2]
    h   = boxes[3]
    
    xmin = xc - w/2
    ymin = yc - h/2
    
    xmax = xmin + w
    ymax = ymin + h
    
    return (xmin, ymin,xmax, ymax)

def center_form(boxes):
    """ Convert prior boxes to (cx, cy, w, h)
    """
    xmin = boxes[0]
    ymin = boxes[1]
    
    xmax = boxes[2]
    ymax = boxes[3]
    
    w  = xmax - xmin
    h  = ymax - ymin
    
    xc = xmin + w/2
    yc = ymin + h/2
    return (xc, yc, w, h)

def intersect(box_a, box_b):
#    box_a = np.array(box_a)
#    box_b = np.array(box_b)

    max_x = max(0, max(box_a[:,0], box_b[:,0]))
    max_y = max(0, max(box_a[:,1], box_b[:,1]))
    
    min_x = min(min(box_a[:,2], box_b[:,2]), 1)
    min_y = min(min(box_a[:,3], box_b[:,3]), 1)
    
    width  = min_x - max_x
    height = min_y - max_y
    
    intersection = width * height
    
    return intersection[0] # Areas of intersection

def jaccard(box_a, box_b):
    
    intersection = intersect(box_a, box_b)
    
    # area_box_a = (xmax - xmin) * (ymax - ymin)
    area_box_a  = (box_a[:,2] - box_a[:,0]) * (box_a[:,3] - box_a[:,1])
    
    area_box_b  = (box_b[:,2] - box_b[:,0]) * (box_b[:,3] - box_b[:,1])
    
    union       = area_box_a + area_box_b - intersection
    
    iou          = intersection/ union
    return iou[0]

def match(truths      = None, 
          labels     = None, 
          priors     = None, 
          variance   = None, 
          threshold  = 0.5, 
          idx        = None,
          loc_t      = None,
          conf_t     = None):
    """
    Match each prior (or anchor) box with the ground truth box of the ighest jaccard overlap, 
    encode the bounding boxes, then return the matched indices correspoding to both confidence 
    and location predictions.
    
    Arguments:
        threshold: (float) The overlap threshold used when matching boxes
        truth    : (tensor) Ground truth boxes, sahep [num_obj, num_priors]
        priors   : (tensor)  Prior boxes from prior boxes layers, shape [num_prioirs, 4]
        variance : (tensor) Variance corresponding to each prioir coordinate, shape [num_priors, 4]
        
        labels   : (tensor) All the class label for the image, shape : [num_obj]
        
        loc_t    : (tensor) Tensor to be filled with encoded location targets
        conf_t   : (tensor) Tensor to be filled with ematched indices for conf preds.
        idx     : (int) current batch index
        
    Returns:
        The match indices corresponding to 
            1) location 
            2) cofidence predcition
    """
    overlaps = jaccard(truths, point_form(priors))
    
    return overlaps

def non_maximum_supression():
    pass


