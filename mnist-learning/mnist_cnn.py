
"""
使用 卷积多层神经网络
"""


import tensorflow as tf

import MNIST_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

session = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeHolder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784,10])
b = tf.Variable(tf.zeros[10])

session.run(tf.global_variables_initializer())

# 预测值
y = tf.nn.softmax(tf.matmul(x,W)+b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
