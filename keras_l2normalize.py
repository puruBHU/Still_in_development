#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 18:26:29 2019

@author: Purnendu Mishra
"""

from keras import backend as K
from keras.enginer.topology import InputSpec
from keras.layers import Layer


class L2Norm(Layer):
    
    def __init__(self, scale, axis):
        self.channel_axis = axis
        self.scale        = scale
        
    def build(self):
         super(L2Norm, self).build(0)
     
    def call(self, x):
        return self.gamma * K.l2_normalize(x, axis = self.channel_axis) 
    
#    def compute_output_shape(self, input_shape):
#        pass
