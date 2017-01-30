"""
MNIST 训练集
train-images-idx3-ubyte.gz  训练图像数据60000个
train-labels-idx1-ubyte.gz  训练图像数据标签60000个
t10k-images-idx3-ubyte.gz   测试图像数据10000个
t10k-labels-idx1-ubyte.gz   测试图像数据标签10000个
"""


import tensorflow as tf
import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

# 输入向量
x = tf.placeholder('float',[None,784])

# 感知机参数
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# softmax 激活函数 output layer
y = tf.nn.softmax(tf.matmul(x,W)+b)

# 交叉熵
