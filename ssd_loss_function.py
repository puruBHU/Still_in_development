#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 23:57:27 2019

@author: Purnendu Mishra
"""
from keras import backend as K
import tensorflow as tf
from utility import match, gather, log_sum_exp
import numpy as np
from keras.utils import to_categorical



class SSDLoss(object):
    
    def __init__(self, anchors,
                 threshold, 
                 num_classes = 21, 
                 alpha = 1.0,
                 neg_pos = 3,
                 variance = [0.1, 0.2]):
        
        self.anchors   = anchors
        self.threshold = threshold
        self.alpha     = alpha
        self.num_classes = num_classes
        self.variance    = variance
        self.negpos_ratio = neg_pos
        
    def smoothL1Loss(self,  y_true, y_pred):
        return tf.losses.huber_loss(labels = y_true, predictions=y_pred)
    
    def classificationLoss(self, y_true, y_pred):
        return K.categorical_crossentropy(target = y_true, output = y_pred)
    
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
#        print(positives.shape)
        
        pos_idx = positives.reshape(positives.shape[0], positives.shape[1], 1).repeat(4, axis=-1)
        
        loc_p = loc_data[pos_idx].reshape(-1,4)
        loc_t = loc_t[pos_idx].reshape(-1,4)
#        print(loc_p.shape)
#        print(loc_t.shape)
        
       
        loc_t = K.cast_to_floatx(loc_t)
        loc_p = K.cast_to_floatx(loc_p)
        loc_loss  = self.smoothL1Loss(y_true = loc_t, y_pred = loc_p)
        
        batch_conf = conf_data.reshape(-1, self.num_classes)
        index      = conf_t.reshape(-1, 1).astype('int')
        
        loss_c     = log_sum_exp(batch_conf) - gather(batch_conf, 1, index)
        
        # Hard Negative Mining
        loss_c     = loss_c.reshape(batch_size, -1)
        loss_c[positives] = 0
        
        loss_idx = np.argsort(loss_c, axis = -1)
        loss_idx = np.flip(loss_idx,  axis = -1)
        
        idx_rank = np.argsort(loss_idx, axis=1)
        
        num_neg = np.clip(self.negpos_ratio * num_positives, a_min = None, a_max = positives.shape[1] - 1)
        num_neg = num_neg.repeat(num_priors, axis=-1)
        
        
        neg = idx_rank < num_neg
       
        pos_idx = positives.reshape(positives.shape[0], positives.shape[1], 1).repeat(self.num_classes, axis=-1)
        neg_idx = neg.reshape(neg.shape[0], neg.shape[1], 1).repeat(self.num_classes, axis=-1)
        
#        print(pos_idx.shape)
#        print(neg_idx.shape)
        
        # Predicted confidence
        conf_p = conf_data[pos_idx + neg_idx]
        conf_p = conf_p.reshape(-1, self.num_classes)
        
        # Transform target confidence in one-hot form
        
        conf_t = to_categorical(conf_t, self.num_classes)
        conf_target = conf_t[pos_idx + neg_idx]
        
#        conf_p = K.cast_to_floatx(conf_p)
#        conf_target = K.cast_to_floatx(conf_target)
        
        loss_confidence = self.classificationLoss(y_true = conf_target, y_pred = conf_p )
        

  
#        N = None
#        
#        loss = 1/N * (conf_loss + self.alpha * loc_loss)
#        
#        if N == 0:
#            loss = 0
        
        return loc_loss, loss_confidence
        

