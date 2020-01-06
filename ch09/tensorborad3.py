import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(123)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

X = tf.compat.v1.placeholder(tf.float32, [None, 2], name='x-input')
Y = tf.compat.v1.placeholder(tf.float32, [None, 1], name='y-input')

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random.normal([2,2]), name='weight1')
    b1 = tf.Variable(tf.random.normal([2]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1)+b1)

    # layer1 에 대한 기록
    W1_hist = tf.compat.v1.summary.histogram('weight1', W1)
    b1_hist = tf.compat.v1.summary.histogram('bias1', b1)
    layer1_hist = tf.compat.v1.summary.histogram('layer1', layer1)

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random.normal([2,1]), name='weight2')
    b2 = tf.Variable(tf.random.normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2)+b2)

    # layer2 에 대한 기록
    W2_hist = tf.compat.v1.summary.histogram('weight2', W2)
    b2_hist = tf.compat.v1.summary.histogram('bias2', b2)
    hypothesis_hist = tf.compat.v1.summary.histogram('hypothesis', hypothesis)




