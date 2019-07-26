#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 23:57:27 2019

@author: Purnendu Mishra
"""
from keras import backend as K
import tensorflow as tf


class SSDLoss(object):
    
    def __init__(self, anchors, threshold):
        self.anchors   = anchors
        self.threshold = anchors
        
    def smoothL1Loss(self):
        pass
    
    def ClassificationLoss(self):
        pass
    
    def ComputeLoss(self, y_true, y_pred):
        pass

