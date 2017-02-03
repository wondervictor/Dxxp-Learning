"""
Network
"""

import numpy as np
import tensorflow as tf
import random



class NeuralNetwork:

    def __init__(self, sizes):
        self.numLayers = len(sizes)
        # 1维向量
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]


    # sigmoid激活函数
    def sigmoid(self,z):
        return 1.0/(1+np.exp(-z))

    # 有点问题
    def feedForward(self, a):
        """输入一个向量a 返回sigmoid(-w*a+b)"""
        for w,b in zip(self.weights, self.biases):
            output = sigmoid(np.dot(w,a)+b)
        return output

    # 随机梯度下降
    def stochasticGradientDescent(self, training_data, epochs, mini_batch_size, learning_rate,test_data=None):
        """
        Train Network using mini_batch Gradient-Descent
        "training_data" a list of (x,y) where x respresents the input and y respresents the ideal output
        """
        if test_data:
            numTest = len(test_data)

        numTrain = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0,n,mini_batch_size)]
                for mini_batch in mini_batches:
                    pass
                if test_data:
                    print("Epoch ")
                else:
                    print("Epoch ")
