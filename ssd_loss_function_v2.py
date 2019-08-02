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
               negpos_ratio = 3,
               num_classes  = 21,
               alpha        = 1.0
               ):
    
    def SSDLoss(y_true, y_pred):

        conf_data =  y_pred[:,:, 4:]
        loc_data  =  y_pred[:,:, :4]
        
        conf_true =  y_true[:,:,-1]
        loc_true  =  y_true[:,:,:-1]
       
        #Since y_pred is list, batch size will
        num_priors = anchors.shape[0]
       
        positives     = K.greater(conf_true , 0)
#        pos           = K.cast(positives, dtype = 'int16')
        
#        num_positives = K.sum(pos, axis = 1, keepdims = True)
        
        pos_shape = K.shape(positives)
        
        pos_idx = K.reshape(positives, shape = (pos_shape[0], pos_shape[1], 1))
        pos_idx = K.repeat_elements(pos_idx, 4, axis = -1)

        
#        loc_p = loc_data[pos_idx]
#        loc_t = loc_true[pos_idx]
        
#        loc_p = K.reshape(loc_p, shape = (-1,4))
#        loc_t = K.reshape(loc_t, shape= (-1, 4))

#        loc_loss  = smoothL1Loss(y_true = loc_t, y_pred = loc_p)
        
#        batch_conf = conf_data.reshape(-1, num_classes)
#        index      = conf_t.reshape(-1, 1).astype('int')
#        
#        loss_c     = log_sum_exp(batch_conf) - gather(batch_conf, 1, index)
#        
#        # Hard Negative Mining
#        loss_c     = loss_c.reshape(batch_size, -1)
#        loss_c[positives] = 0
#        
#        loss_idx = np.argsort(loss_c, axis = -1)
#        loss_idx = np.flip(loss_idx,  axis = -1)
#        
#        idx_rank = np.argsort(loss_idx, axis=1)
#        
#        num_neg = np.clip(negpos_ratio * num_positives, a_min = None, a_max = positives.shape[1] - 1)
#        num_neg = num_neg.repeat(num_priors, axis=-1)
#        
#        
#        neg = idx_rank < num_neg
#       
#        pos_idx = positives.reshape(positives.shape[0], positives.shape[1], 1).repeat(num_classes, axis=-1)
#        neg_idx = neg.reshape(neg.shape[0], neg.shape[1], 1).repeat(num_classes, axis=-1)
#        
#
#        # Predicted confidence
#        conf_p = conf_data[pos_idx + neg_idx]
#        conf_p = conf_p.reshape(-1, num_classes)
#        
#        # Transform target confidence in one-hot form
#        
#        conf_t = to_categorical(conf_t, num_classes)
#        conf_target = conf_t[pos_idx + neg_idx]
#        conf_target = conf_target.reshape(-1, num_classes)
#
#        conf_p = K.cast_to_floatx(conf_p)
#        conf_target = K.cast_to_floatx(conf_target)
#        
#        loss_confidence = classificationLoss(y_true = conf_target, y_pred = conf_p)
#        
#
#  
#        N = np.sum(num_positives)
#        
#        loss = 1/N * (loss_confidence + alpha * loc_loss)
        
      
        return pos_idx
    return SSDLoss