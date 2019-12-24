import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# 5행 3열
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96.,98.,100.],
          [73.,66.,70.]]

# 5행 1열
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]


X = tf.compat.v1.placeholder(tf.float32, shape=[None, 3]) # 데이터

Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) # 결과값

# X의 열을 W의 행과 같게 Y의 열이 W의 열의 개수와 일치
W = tf.Variable(tf.random.normal([3,1]),name='weight')
# b의 개수는 W의 열의 개수와 일치
b = tf.Variable(tf.random.normal([1]), name='bias')
hypothesis = tf.matmul(X,W)
cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(2000):
    cost_, hy_, t_  = sess.run([cost,hypothesis,train], feed_dict={X:x_data, Y:y_data})
    if i%20 == 0:
        print('cost :', cost_, "pridect:\n", hy_)

sess.close()


