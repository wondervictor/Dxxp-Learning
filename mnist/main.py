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
y_ = tf.nn.softmax(tf.matmul(x,W)+b)

# y 的输入正确值
y = tf.placeholder("float",[None, 10])

# 交叉熵
cross_entropy = -tf.reduce_sum(y*tf.log(y_))

# 使用梯度下降算法最小化交叉熵

train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init_op = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init_op)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    aspdd = sess.run(train,{x:batch_xs,y:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print(sess.run(accuracy,{x:mnist.test.images,y:mnist.test.labels}))
