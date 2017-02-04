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
        Train Network using mini_batchn  Gradient-Descent
        "training_data" a list of (x,y) where x respresents the input and y respresents the ideal output
        """
        if test_data:
            numTest = len(test_data)

        numTrain = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch,learning_rate)
                if test_data:
                    print("Epoch %s :  %s / %s" %(j,self.evaluate(test_data),numTest))
                else:
                    print("Epoch %s completed" %(j))

    def update_mini_batch(self, mini_batch, learning_rate):
        """
        update the weights and bias by training the model
        ``mini_batch``  is a list of tuple (x,y)
        """
        b = [np.zeros(b.shape)]
        weight = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_b, delta_weight = self.backProp(x,y)
            b = [nb+dnb for nb,dnb in zip(b,delta_b)]
            weight = [nw+dnw for nw, dnw in zip(weight,delta_weight)]
        self.weights = [w - (learning_rate/len(mini_batch))*nw for w, nw in zip(self.weights,weight)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb for b, nb in zip(self.biases,b)]

    def backProp(self,x,y):
        """
        return the tuple(theta_b, theta_weight) represents the gradient for the cost function C
        theta_b, theta_weight are layer-by-layer lists of numpy arrays
        """

        theta_b = [np.zeros(b.shape) for b in self.biases]
        theta_weight = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b,w in zip(self.biases, self.weights):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivate(activations[-1],y) * self.sigmoid_prime(zs[-1])

        theta_b[-1] = delta
        theta_weight[-1] = np.dot(delta,activations[-2].transpose())

        for l in range(2, self.numLayers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta)*sp
            theta_b[-l] = delta
            theta_weight[-l] = np.dot(delta,activations[-l-1].transpose())
        return (theta_b, theta_weight)

    # cost函数微分
    def cost_derivate(self, output_activations, y):
        return (output_activations-y)


    def sigmoid_prime(self,z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))\


    # 模型评测
    def evaluate(self,test_data):
        test_result = [(np.argmax(self.feedForward(x)),y) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in test_result)
