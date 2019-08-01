#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:03:18 2019

@author: Purnendu Mishra
"""

from keras import backend as K
import tensorflow as tf
from utility_keras_form import match, gather, log_sum_exp
import numpy as np
from keras.utils import to_categorical

def smoothL1Loss(target, output):
    return tf.losses.huber_loss(labels = target, predictions=output)
    
def classificationLoss(target, output):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=output)
    return K.mean(loss)

def CustomLoss(anchors      = None, 
               threshold    = 0.6, 
               variance     = [0.1, 0.2],
               negpos_ratio = 3,
               num_classes  = 21,
               alpha        = 1.0,
               batch_size   = 4):
    
    def SSDLoss(y_true, y_pred):

        conf_data =  y_pred[:,:, 4:]
        loc_data  =  y_pred[:,:, :4]
       
        #Since y_pred is list, batch size will
        num_priors = anchors.shape[0]
       
        print(type(y_pred))
        print(type(y_true))
        
        loc_t  = np.zeros(shape = (batch_size, num_priors, 4), dtype=np.float32)
        conf_t = np.zeros(shape = (batch_size, num_priors), dtype=np.float32)
       
        for idx in range(batch_size):
            true_loc       = y_true[idx][:,1:]
            true_class_id  = y_true[idx][:,0]

            match(truths    = true_loc,  
                  labels    = true_class_id, 
                  loc_t     = loc_t, 
                  conf_t    = conf_t,
                  variance  = variance,
                  threshold = threshold,
                  priors    = anchors)
         
        positives     = conf_t > 0
        num_positives = np.sum(positives, axis = 1, keepdims = True)

        
        pos_idx = positives.reshape(positives.shape[0], positives.shape[1], 1).repeat(4, axis=-1)
        
        loc_p = loc_data[pos_idx].reshape(-1,4)
        loc_t = loc_t[pos_idx].reshape(-1,4)

            
        loc_t = K.cast_to_floatx(loc_t)
        loc_p = K.cast_to_floatx(loc_p)
        loc_loss  = smoothL1Loss(y_true = loc_t, y_pred = loc_p)
        
        batch_conf = conf_data.reshape(-1, num_classes)
        index      = conf_t.reshape(-1, 1).astype('int')
        
        loss_c     = log_sum_exp(batch_conf) - gather(batch_conf, 1, index)
        
        # Hard Negative Mining
        loss_c     = loss_c.reshape(batch_size, -1)
        loss_c[positives] = 0
        
        loss_idx = np.argsort(loss_c, axis = -1)
        loss_idx = np.flip(loss_idx,  axis = -1)
        
        idx_rank = np.argsort(loss_idx, axis=1)
        
        num_neg = np.clip(negpos_ratio * num_positives, a_min = None, a_max = positives.shape[1] - 1)
        num_neg = num_neg.repeat(num_priors, axis=-1)
        
        
        neg = idx_rank < num_neg
       
        pos_idx = positives.reshape(positives.shape[0], positives.shape[1], 1).repeat(num_classes, axis=-1)
        neg_idx = neg.reshape(neg.shape[0], neg.shape[1], 1).repeat(num_classes, axis=-1)
        

        # Predicted confidence
        conf_p = conf_data[pos_idx + neg_idx]
        conf_p = conf_p.reshape(-1, num_classes)
        
        # Transform target confidence in one-hot form
        
        conf_t = to_categorical(conf_t, num_classes)
        conf_target = conf_t[pos_idx + neg_idx]
        conf_target = conf_target.reshape(-1, num_classes)

        conf_p = K.cast_to_floatx(conf_p)
        conf_target = K.cast_to_floatx(conf_target)
        
        loss_confidence = classificationLoss(y_true = conf_target, y_pred = conf_p)
        

  
        N = np.sum(num_positives)
        
        loss = 1/N * (loss_confidence + alpha * loc_loss)
        
      
        return loss
    return SSDLoss