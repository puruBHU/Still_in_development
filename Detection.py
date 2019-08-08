#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:24:14 2019

@author: Purnendu Mishra

"""
import numpy as np
from nms import nms

class Detect(object):
    
    def __inti__(self, 
                 prediction, 
                 num_classes, 
                 priors):
        self.loc_data = prediction[:,:4]
        self.conf_data = prediction[:.4:]
        
        self.num_classes = num_classes
        self.priors      = priors
    
    def Detection(self):
        pass

