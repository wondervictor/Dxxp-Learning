import tensorflow as tf
import numpy as np


# 线性
xdata = np.random.rand(100).astype(np.float32)
ydata = xdata * 0.1 + 0.3


# 设置权重和感知机参数
weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))
y = weight * xdata + b

# 损失
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.7)
train = optimizer.minimize(loss)


# 初始化变量
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

for step in range(201):
    """ 20迭代后进行输出 """
    session.run(train)
    if step % 20 == 0:
        print(step, session.run(weight),session.run(b))
