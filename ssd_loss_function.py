#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 23:57:27 2019

@author: Purnendu Mishra
"""
from keras import backend as K
import tensorflow as tf


class SSDLoss(object):
    
    def __init__(self, anchors, threshold, alpha):
        self.anchors   = anchors
        self.threshold = anchors
        self.alpha     = alpha
        
    def smoothL1Loss(self,  y_true, y_pred):
        pass
    
    def ClassificationLoss(self, y_true, y_pred):
        pass
    
    def ComputeLoss(self, y_true, y_pred):
        
        conf_loss = self.ClassificationLoss(y_true, y_pred)
        loc_loss  = self.smoothL1Loss(y_true, y_pred)
        
        N = None
        
        loss = 1/N * (conf_loss + self.alpha * loc_loss)
        
        return loss
        

