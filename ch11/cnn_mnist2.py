import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow_core.examples import input_data

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(123)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
learning_rate = 0.001
traing_epochs = 15
batch_size = 100
X = tf.compat.v1.placeholder(tf.float32, [None, 784])

# 이미지 갯수, 가로, 세로, 색
X_img = tf.reshape(X, [-1,28,28,1])
Y = tf.compat.v1.placeholder(tf.float32,[None, 10])
# 필터의 크기3,3,컬러, 필터수
W1 = tf.Variable(tf.random.normal([3,3,1,32], stddev=0.01))

# L1 : conv2d -> 사이즈 동일(SAME).. ?, 28, 28, 32  -> 필터수가 32 이면 결과도 32개
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1) #

# L1 : 사이즈가 strides 2,2 때문에 1/2 로 줄어든다. ?, 14,14, 32
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
W2 = tf.Variable(tf.random.normal([3,3,32,64], stddev=0.01))

# L2 :  ?, 14, 14, 64
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
# L2 : ? 7, 7, 64
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

L2 = tf.reshape(L2, [-1, 7 * 7 * 64]) # 7 * 7 * 64 = 3136

W3 = tf.compat.v1.get_variable('w3', shape=[3136, 10], initializer=tf.compat.v1.glorot_normal_initializer())

b = tf.Variable(tf.random.normal([10]))
hypothesis = tf.matmul(L2, W3) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
train = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
print('시작')
for epoch in range(traing_epochs):
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X:batch_xs, Y:batch_ys}
        c_, _ = sess.run([cost,train], feed_dict=feed_dict)
    print(epoch,"회 작업")
correction_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))
print('정확도 :', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
sess.close()









