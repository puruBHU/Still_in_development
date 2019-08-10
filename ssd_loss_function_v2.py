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
    return tf.compat.v1.losses.huber_loss(labels = target, predictions=output)
    
def classificationLoss(target, output):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=output)
    return K.mean(loss)

def CustomLoss(anchors      = None, 
               negpos_ratio = 3,
               num_classes  = 21,
               alpha        = 1.0
               ):
    
    def SSDLoss(y_true, y_pred):

        conf_pred =  y_pred[:,:, 4:] # Predicted confidence
        loc_pred  =  y_pred[:,:, :4] # Predicted location
        
        conf_true =  y_true[:,:,4:]  # Ground truth confidence
        loc_true  =  y_true[:,:,:4]  # Ground truth location
       
        conf_true_ = K.argmax(conf_true, axis=-1)
        #Since y_pred is list, batch size will
        num_priors    = anchors.shape[0]
       
        positives     = K.greater(conf_true_ , 0)
        pos           = K.cast(positives, dtype = 'float32')
        
        num_positives = K.sum(pos, axis = 1, keepdims = True)
        
        pos_shape     = K.shape(positives)
        
        pos_idx = K.reshape(positives, shape = (pos_shape[0], pos_shape[1], 1))
        pos_idx = K.repeat_elements(pos_idx, 4, axis = -1)

        # Masking of the tensor
        loc_p = K.tf.boolean_mask(loc_pred, pos_idx)
        loc_t = K.tf.boolean_mask(loc_true, pos_idx)
#        loc_t = loc_true[pos_idx]
        
        loc_p = K.reshape(loc_pred, shape = (-1,4))
        loc_t = K.reshape(loc_true, shape= (-1, 4))
        
        # localization loss
        loc_loss  = smoothL1Loss(target = loc_t, output = loc_p)
        
        # Classification loss
        ##  Positive
        # masking the Tensor
        conf_pred_pos = K.tf.boolean_mask(conf_pred, pos_idx)
        conf_true_pos = K.tf.boolean_mask(conf_true, pos_idx)
        
        class_loss = classificationLoss(target = conf_true, output=conf_pred)
       
        # # Compute max conf across batch for hard negative mining
#        batch_conf = K.reshape(conf_pred, shape=(-1, num_classes))
        
        N = K.sum(num_positives)
        N = K.maximum(1.0, N)
        
       
        loss =  class_loss + alpha * loc_loss
        loss /= N
      
        return loss
    return SSDLoss