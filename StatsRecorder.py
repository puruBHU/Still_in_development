#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 09:53:34 2018

@author: vlsilab
"""

import numpy as np

keepdims= False
axis = (0,1,2)

class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=axis, keepdims=keepdims)
            self.std  = data.std(axis=axis, keepdims=keepdims)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=axis, keepdims=keepdims)
            newstd  = data.std(axis=axis, keepdims=keepdims)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = np.sqrt(self.std)

            self.nobservations += n
            
            
if __name__=='__main__':
    rs = np.random.RandomState(323)

    mystats = StatsRecorder()
    
    # Hold all observations in "data" to check for correctness.
    ndims = 42
    data = np.empty((0, ndims))
    
    for i in range(1000):
        nobserv = rs.randint(10,101)
        newdata = rs.randn(nobserv, ndims)
        data = np.vstack((data, newdata))
    
        # Update stats recorder object
        mystats.update(newdata)
    
        # Check stats recorder object is doing its business right.
        assert np.allclose(mystats.mean, data.mean(axis=0))
        assert np.allclose(mystats.std, data.std(axis=0))