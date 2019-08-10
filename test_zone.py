#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:58:26 2019

@author: puru
"""
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
##********************************************************
## For GPU
#
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.50
set_session(tf.Session(config=config))

#
##********************************************************

from keras import backend as K


from SSD_generate_anchors import generate_ssd_priors

from CustomDataLoaderv3 import DataAugmentor

from utility import *
from pathlib import Path
import collections
from ssd300_model import SSD300
from skimage.io import imread, imshow
import cv2
import numpy as np
#from ssd_loss_function import SSDLoss
from ssd_loss_function_v2 import CustomLoss
from keras.losses import categorical_crossentropy


session = K.get_session()

 
#root                = Path.home()/'data'/'VOCdevkit'/'VOC2007'
root                 = Path.home()/'Documents'/'DATASETS'/'VOCdevkit'/'VOC2007'
voc_2007_datafile  = root/'ImageSets'/'Main'/'train.txt'

voc_2007_images      = root/'JPEGImages'
voc_2007_annotations = root/'Annotations'


SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])

Spec = collections.namedtuple('Spec', ['feature_map_size', 'shrinkage', 'box_sizes', 
                                       'aspect_ratios'])

# the SSD orignal specs
specs = [
    Spec(38, 8, SSDBoxSizes(30, 60), [2]),
    Spec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
    Spec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
    Spec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
    Spec(3, 100, SSDBoxSizes(213, 264), [2]),
    Spec(1, 300, SSDBoxSizes(264, 315), [2])
]

priors = generate_ssd_priors(specs)

batch_size = 4

testloader = DataAugmentor()
data       = testloader.flow_from_directory(root        = root,
                                            data_file   = voc_2007_datafile,
                                            target_size = 300,
                                            batch_size  = batch_size,
                                            shuffle     = True,
                                            num_classes = 21,
                                            priors      = priors
                                            )

sample = data[0]

images, targets = sample

images /= 255.0

def preprocess_image(image_path):
    image = imread(image_path)
    img = cv2.resize(image, (300, 300), interpolation = cv2.INTER_CUBIC)
    img = np.expand_dims(img,axis = 0)
    img = np.float32(img)
    img /= 255.0
    return image, img

_, img = preprocess_image('000018.jpg')

model = SSD300(input_shape=(300,300, 3), num_classes=21)

prediction = model.predict(images)

SSDLoss = CustomLoss(anchors   = priors, 
                     alpha     = 1.0
                     )

loss = SSDLoss(y_true = targets, y_pred = prediction)

print(K.eval(loss))
#print(K.eval(N))


