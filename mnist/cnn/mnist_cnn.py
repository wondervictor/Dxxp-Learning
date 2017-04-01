
"""
使用 卷积多层神经网络
"""


import tensorflow as tf
import input_data


# 权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)



# 卷积与pool
def conv(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

session = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# 第一层卷积
# 权重
weight_conv1 = weight_variable([5,5,1,32])
# 偏置
b_conv1 = bias_variable([32])

# 将图像数据向量转化为 图像矩阵 28*28 方便使用卷积变换
image_mat = tf.reshape(x,[-1,28,28,1])

# ReLU激活
hConv1 = tf.nn.relu(conv(image_mat,weight_conv1)+b_conv1)
# 卷积池化
hPool1 = max_pool(hConv1)

# 第二层卷积
weight_conv2 = weight_variable([5,5,32,64])

b_conv2 = bias_variable([64])

hConv2 = tf.nn.relu(conv(hPool1,weight_conv2)+b_conv2)

hPool2 = max_pool(hConv2)

weight_3 = weight_variable([49*64, 1024])
b_3 = bias_variable([1024])

hPoolFlat = tf.reshape(hPool2,[-1,49*64])
h_3 = tf.nn.relu(tf.matmul(hPoolFlat,weight_3)+b_3)

# Drop out 防止过拟合
keep_prob = tf.placeholder("float")
h_drop = tf.nn.dropout(h_3,keep_prob)

# output layer
weight_4 = weight_variable([1024,10])
b_4 = bias_variable([10])

y = tf.nn.softmax(tf.matmul(h_drop,weight_4)+b_4)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))


train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1));
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
session.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step:%d training accuracy: %g" %(i,train_accuracy))
        session.run(train,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})


print("step:%d training accuracy: %g" %(i,train_accuracy))
