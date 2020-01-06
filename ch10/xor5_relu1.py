
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
W1 = tf.compat.v1.get_variable( name='weight1',shape=[2,10]) # w 의 가지수를 5배로 늘렸을때
b1 = tf.compat.v1.get_variable( name='bias1', shape=[10])
layer1 = tf.nn.relu(tf.matmul(X,W1) +b1) # 1차

W2 = tf.compat.v1.get_variable( name='weight2',shape=[10,10]) #
b2 = tf.compat.v1.get_variable( name='bias2', shape=[10])
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2) # 2차

W3 = tf.compat.v1.get_variable( name='weight3',shape=[10,10]) #
b3 = tf.compat.v1.get_variable( name='bias3', shape=[10])
layer3 = tf.nn.relu(tf.matmul(layer2,W3) +b3) # 3차

W4 = tf.compat.v1.get_variable( name='weight4',shape=[10,10]) #
b4 = tf.compat.v1.get_variable( name='bias4', shape=[10])
layer4 = tf.nn.relu(tf.matmul(layer3,W4) +b4) # 4차

W5 = tf.compat.v1.get_variable( name='weight5',shape=[10,10]) #
b5 = tf.compat.v1.get_variable( name='bias5', shape=[10])
layer5 = tf.nn.relu(tf.matmul(layer4,W5) +b5) # 5차

W6 = tf.compat.v1.get_variable( name='weight6',shape=[10,10]) #
b6 = tf.compat.v1.get_variable( name='bias6', shape=[10])
layer6 = tf.nn.relu(tf.matmul(layer5,W6) +b6) # 6차

W7 = tf.compat.v1.get_variable( name='weight7',shape=[10,10]) #
b7 = tf.compat.v1.get_variable( name='bias7', shape=[10])
layer7 = tf.nn.relu(tf.matmul(layer6,W7) +b7) # 7차

W8 = tf.compat.v1.get_variable( name='weight8',shape=[10,1]) #
b8 = tf.compat.v1.get_variable( name='bias8', shape=[1])
hypothesis = tf.sigmoid(tf.matmul(layer7, W8) + b8) # 8차


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