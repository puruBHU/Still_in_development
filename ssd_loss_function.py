#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 23:57:27 2019

@author: Purnendu Mishra
"""
from keras import backend as K
import tensorflow as tf
from utility import match
import numpy as np
from keras.utils import to_categorical



class SSDLoss(object):
    
    def __init__(self, anchors,
                 threshold, 
                 num_classes = 21, 
                 alpha = 1.0,
                 variance = [0.1, 0.2]):
        
        self.anchors   = anchors
        self.threshold = threshold
        self.alpha     = alpha
        self.num_classes = num_classes
        self.variance    = variance
        
    def smoothL1Loss(self,  y_true, y_pred):
        return tf.losses.huber_loss(labels = y_true, predictions=y_pred)
    
    def ClassificationLoss(self, y_true, y_pred):
        return K.sparse_categorical_crossentropy(target = y_true, output = y_pred)
    
    def ComputeLoss(self, y_true, y_pred):
        
        conf_data, loc_data =  y_pred
        
#        print(conf_data.shape)
#        print(loc_data.shape)
        #Since y_pred is list, batch size will
        batch_size, num_priors, _ = conf_data.shape
        
  
        
        loc_t  = np.zeros(shape = (batch_size, num_priors, 4), dtype=np.float32)
        conf_t = np.zeros(shape = (batch_size, num_priors), dtype=np.float32)
       
        for idx in range(batch_size):
            true_loc       = y_true[idx][:,1:]
            true_class_id  = y_true[idx][:,0]
#            print(true_class_id)
            
            match(truths    = true_loc,  
                  labels    = true_class_id, 
                  loc_t     = loc_t, 
                  conf_t    = conf_t,
                  variance  = self.variance,
                  threshold = self.threshold,
                  priors    = self.anchors)
         
        positives     = conf_t > 0
        num_positives = np.sum(positives, axis = 1, keepdims = True)
        
        pos_idx = positives.reshape(positives.shape[0], positives.shape[1], 1).repeat(4, axis=-1)
        
        loc_p = loc_data[pos_idx].reshape(-1,4)
        loc_t = loc_t[pos_idx].reshape(-1,4)
#        print(loc_p.shape)
#        print(loc_t.shape)
        
       
        loc_t = K.cast_to_floatx(loc_t)
        loc_p = K.cast_to_floatx(loc_p)
        loc_loss  = self.smoothL1Loss(y_true = loc_t, y_pred = loc_p)
        
        batch_conf = conf_data.reshape(-1, self.num_classes)
        print(batch_conf.shape)
  
#        N = None
#        
#        loss = 1/N * (conf_loss + self.alpha * loc_loss)
#        
#        if N == 0:
#            loss = 0
        
        return loc_loss, conf_t
        

