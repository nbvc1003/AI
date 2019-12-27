import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, :-1] # 마지막열 제외 전부
y_data = xy[:, [-1]] # 마지막 열만

X = tf.compat.v1.placeholder(tf.float32, shape=[None,8])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

# [X의 열의 갯수, Y의 열의 갯수]
W = tf.Variable(tf.random.normal([8, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y*tf.math.log(hypothesis) +
                       (1 - Y)*tf.math.log(1-hypothesis))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# hypothesis 값에 따라서 0 or 1
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    feed = {X:x_data, Y:y_data}
    for i in range(10001):
        sess.run(train, feed_dict=feed)
        if i % 200 == 0:
            print(i,sess.run(cost,feed_dict=feed))
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("hypothesis :", h," 예측: ",p, " 정확도:", a)






