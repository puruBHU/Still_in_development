#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 23:57:27 2019

@author: Purnendu Mishra
"""
from keras import backend as K
import tensorflow as tf
from utility import match




class SSDLoss(object):
    
    def __init__(self, anchors,
                 threshold, 
                 num_classes = 21, 
                 alpha = 1.0,
                 variance = [0.1, 0.2]):
        
        self.anchors   = anchors
        self.threshold = anchors
        self.alpha     = alpha
        self.num_classes = num_classes
        self.variance    = variance
        
    def smoothL1Loss(self,  y_true, y_pred):
        return tf.losses.huber_loss(labels = y_true, predictions=y_pred)
    
    def ClassificationLoss(self, y_true, y_pred):
        return K.sparse_categorical_crossentropy(target = y_true, output = y_pred)
    
    def ComputeLoss(self, y_true, y_pred):
        
        #Since y_pred is list, batch size will
        batch_size = len(y_pred)
        num_priors = self.anchors.shape[0]
        
        # Get the class confidence score of prediction and the ground truth
#        y_true_conf = y_true[:,:,0]
#        y_pred_conf = y_pred[:,:,0]
        
        loc_t  = np.zeros(batch_size, num_priors, 4)
        conf_t = np.zeros(batch_size, num_priors)
        
        for idx in range(batch_size):
            true_loc       = y_true[idx][:,1:]
            true_class_id  = y_true[idx][:,0]
            
            match(truths    = true_loc,  
                  labels    = true_class_id, 
                  loc_t     = loc_t, 
                  conf_t    = conf_t,
                  variance  = variance,
                  threshold = self.threshold,
                  priors    = self.anchors)
            
        
        conf_loss = self.ClassificationLoss(y_true, y_pred)
        loc_loss  = self.smoothL1Loss(y_true, y_pred)
        
        N = None
        
        loss = 1/N * (conf_loss + self.alpha * loc_loss)
        
        if N == 0:
            loss = 0
        
        return loss
        

