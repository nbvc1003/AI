import tensorflow as tf
tf.compat.v1.disable_eager_execution()

tf.compat.v1.set_random_seed(1234)

x_data = [[1,2,1],[1,3,2],[1,3,4],[1,5,5],
          [1,7,5],[1,2,5],[1,6,6],[1,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],
          [0,1,0],[0,1,0],[1,0,0],[1,0,0]]

x_test = [[2,1,1],[3,1,2],[3,3,4]]
y_test = [[0,0,1],[0,0,1],[0,0,1]]
X = tf.compat.v1.placeholder(tf.float32, [None, 3])
Y = tf.compat.v1.placeholder(tf.float32, [None, 3])
# X의 혈갯수, Y와열갰수 행열
W = tf.Variable(tf.random.normal([3,3]))
b = tf.Variable(tf.random.normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W)+ b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.math.log(hypothesis), axis=1))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-100).minimize(cost)
prediction = tf.math.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.math.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(201):
        cost_, W_, _, a_ = sess.run([cost, W,train, accuracy], feed_dict={X:x_data,Y:y_data})
        print(i, cost_,W_, a_)
    print('예측 :', sess.run(prediction, feed_dict={X:x_test}))
    print('정확도 :',sess.run(accuracy, feed_dict={X:x_test,Y:y_test}))


