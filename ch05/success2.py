import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
xy = np.loadtxt('train.txt', dtype=np.float32)
x_data = xy[:, :-1]
y_data = xy[:,[-1]]

X = tf.compat.v1.placeholder(tf.float32, shape=[None,3])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])
# [X의 열의 갯수, Y의 열의 갯수]
W = tf.Variable(tf.random.normal([3, 1]))
b = tf.Variable(tf.random.normal([1]))

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y*tf.math.log(hypothesis) +
                       (1 - Y)*tf.math.log(1-hypothesis))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# hypothesis 값에 따라서 0 or 1
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(10001):
    c_, t_ = sess.run([cost, train],
                      feed_dict={X:x_data, Y:y_data})
    if i % 100 == 0:
        print(i,c_)
h, c, a = sess.run([hypothesis, cost, accuracy], feed_dict={X:x_data, Y:y_data})
print("예측 :", h," 비용: ",c, " 정확도:", a)

sess.close()
