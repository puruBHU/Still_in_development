#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:16:15 2018

@author: Purnendu Mishra
"""

import pandas as pd
from CustomDataLoader import DataAugmentor
import matplotlib.pyplot as plt
import numpy as np
from StatsRecorder import StatsRecorder
import os
from pathlib import Path


root =  Path.home()/'Documents'/'DATASETS'/'VOCdevkit'/'VOC2007'
voc_2012_datafile = root/'ImageSets'/'Main'/'train.txt'
voc_2012_images   = root/'JPEGImages'
voc_2012_annotations = root/'Annotations'

#images_path = os.path.join(root, 'Images')
#labels_path  = os.path.join(root, 'train.csv')

target_size = (300,300)

#m = np.array([116.54128, 111.74498, 103.55727], dtype=np.float32)
#s  = np.array( [69.750916, 69.02749,  72.47146], dtype=np.float32)

test_gen = DataAugmentor(normalize=False,
                         mean = m,
                         std  = s)


datagen = test_gen.flow_from_directory(root = root,
                                            data_file=voc_2012_datafile,
                                            target_size=target_size,
                                            batch_size = 32,
                                            shuffle    = False
                                            )
#for i in range(5):
#    image_batch, label_batch = datagen[i] 
#    fig = plt.figure(figsize=(10,10))
#    for j in range(image_batch.shape[0]):
#        
#        ax = plt.subplot(1,3, j + 1)
#        plt.tight_layout()
#        image_array = np.hstack((image_batch[j][:,:,0], label_batch[j][:,:,0]))
#        plt.imshow(image_array.astype('uint8'))
##        print np.unique(image_batch[j])
##        print np.unique(label_batch[j])
##        print image_batch[j].shape
##        print label_batch[j].shape
#    plt.show()

mean = np.zeros(target_size, np.float32) 
std  = np.zeros(target_size, np.float32)

#data = np.empty((0, target_size))

mystats = StatsRecorder()

for i in range(len(datagen)):
    image_batch, label_batch = datagen[i]

    mystats.update(image_batch)


mean = mystats.mean
std = mystats.std

print('mean: ',mean)
print('std: ',std)
  
#hdf5_path = 'skin_hsv_mean_and_std.h5'
#with h5py.File(hdf5_path, 'w') as f:
#    f.create_dataset('mean', target_size, np.float32)
#    f.create_dataset('std', target_size, np.float32)
#    
#    
#    f['mean'][...] = mean[None]
#    f['std'][...] = std[None]