import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]],dtype=np.float32)
X= tf.compat.v1.placeholder(tf.float32)
Y= tf.compat.v1.placeholder(tf.float32)
W = tf.Variable(tf.random.normal([2,1], name='weight'))
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W) +b)
cost = -tf.reduce_mean(Y*tf.math.log(hypothesis)+ (1-Y)* tf.math.log(1-hypothesis))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accurary = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(10001):
        c_, _,w_ = sess.run([cost, train, W], feed_dict={X:x_data, Y:y_data})
        if i% 300 == 0 :
            print('cost:', c_, ' w :',w_)
            # print(i, sess.run(cost, feed_dict={X:x_data,Y:y_data}),
            #       sess.run(W))
    h_, c_, a_ = sess.run([hypothesis, predicted, accurary], feed_dict={X: x_data, Y: y_data})
    print('예측값 : ', h_, ' hypothesis :', h_, ' a :',a_)


# 1레이어 만으로는 정확도가 0.5를 넘지 못한다.


