# -*- coding: utf-8 -*-
# Author: VicCha
# Date: 4/2/2017
# Content: Data Provider

import numpy as np
import struct
import matplotlib.pyplot as plt

def readImageFiles(fileName):
    binFile = open(fileName, 'rb')
    buf = binFile.read()
    index = 0
    # 前四个32位integer为以下参数
    # >IIII 表示使用大端法读取
    magic, numImage, numRows, numCols = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    image_sets = []
    for i in range(numImage):
        images = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        images = np.array(images)
        image_sets.append(image_sets)
    binFile.close()
    return image_sets

def readLabelFiles(fileName):
    binFile = open(fileName, 'rb')
    buf = binFile.read()
    index = 0
    magic, nums = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    labels = struct.unpack_from('>%sB'%nums, buf, index)
    binFile.close()
    labels = np.array(labels)
    print(len(labels))
    return labels

def fetchTraingSet():
    pass


def fetchTestingSet():
    pass


#readFiles("../MNIST_data/train-images-idx3-ubyte")
#readLabelFiles('../MNIST_data/train-labels-idx1-ubyte')
