# -*- coding: utf-8 -*-
# Content: VGG-Net
# Author: Vic Chan
# Date: 4/2/2017

import tensorflow as tf
import numpy as np
import dataprovider

def initializeWeights(shape):
    weight = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weight)

def initializeBias(shape):
    bias = tf.constant(0.1, shape=shape)
    return tf.Variable(bias)


x = tf.placeholder(tf.float, shape=[None, 784])
label = tf.placeholder(tf.float, shape=[None, 10]])
