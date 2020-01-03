import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]],dtype=np.float32)
X= tf.compat.v1.placeholder(tf.float32)
Y= tf.compat.v1.placeholder(tf.float32)

W1 = tf.Variable(tf.random.normal([2,2], name='weight1')) #
b1 = tf.Variable(tf.random.normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X,W1) +b1) # 1차

W2 = tf.Variable(tf.random.normal([2,1], name='weight2'))
b2 = tf.Variable(tf.random.normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2) # 2차

cost = -tf.reduce_mean(Y*tf.math.log(hypothesis)+ (1-Y)* tf.math.log(1-hypothesis))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accurary = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(20001):
        c_, _= sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
        if i% 300 == 0 :
            print('cost:', c_)
            # print(i, sess.run(cost, feed_dict={X:x_data,Y:y_data}),
            #       sess.run(W))
    h_, c_, a_ = sess.run([hypothesis, predicted, accurary], feed_dict={X: x_data, Y: y_data})
    print('예측값 : ', h_, ' hypothesis :', h_, ' a :',a_)


# 1레이어 만으로는 정확도가 0.5를 넘지 못한다.


