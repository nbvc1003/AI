import tensorflow as tf
import numpy as np
# 4레이어 10개w 일경우
tf.compat.v1.disable_eager_execution()
x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]],dtype=np.float32)
X= tf.compat.v1.placeholder(tf.float32)
Y= tf.compat.v1.placeholder(tf.float32)

# 1레이어의 결과를 2개이상의 여러개로 놓고 계산 한다.
# 정확도가 높아 진다.
W1 = tf.Variable(tf.random.normal([2,10], name='weight1')) # w 의 가지수를 5배로 늘렸을때
b1 = tf.Variable(tf.random.normal([10]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X,W1) +b1) # 1차

W2 = tf.Variable(tf.random.normal([10,10], name='weight2')) #
b2 = tf.Variable(tf.random.normal([10]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2) # 2차

W3 = tf.Variable(tf.random.normal([10,10], name='weight3')) # w 의 가지수를 5배로 늘렸을때
b3 = tf.Variable(tf.random.normal([10]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2,W3) +b3) # 1차

W4 = tf.Variable(tf.random.normal([10,1], name='weight4')) #
b4 = tf.Variable(tf.random.normal([1]), name='bias4')
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4) # 2차


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


