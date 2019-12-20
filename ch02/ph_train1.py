import tensorflow as tf
tf.compat.v1.disable_eager_execution()
w = tf.Variable(tf.compat.v1.random_normal([1]), name='weight')
b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')
X = tf.compat.v1.placeholder(tf.float32)
Y = tf.compat.v1.placeholder(tf.float32)
hypothesis = X * w + b
# (에측값 - 실제y) ** 2한 것을 평균
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# train 경사하강법을 따라서 learning_rate만큼 w,b값 변경하여 cost최소
train = tf.compat.v1.train.GradientDescentOptimizer(
    learning_rate = 0.01).minimize(cost)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(3001):
    # X, Y 를 실행할 때 제공 placeholder
    cost_,w_,b_,t_ = sess.run([cost, w, b, train],
             feed_dict={X:[1,2,3,4,5],Y:[2.1,3.1,4.1,5.1,6.1]})
    if i %20 == 0:
        print(i, cost_, w_, b_)
sess.close()