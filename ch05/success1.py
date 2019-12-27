import tensorflow as tf

tf.compat.v1.disable_eager_execution()
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]  # 6 * 2
y_data = [[0],[0],[0],[1],[1],[1]] # 6*1

X = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random.normal([2,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# 구한 값을 0 ~ 1사이로 변환
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# sigmoid 값이 0과 1로 울퉁불퉁해서 매끄럽게 변환 한다.
cost = -tf.reduce_mean(
                        Y*tf.compat.v1.log(hypothesis) +  # 1일때 부분   
                       (1-Y)*tf.compat.v1.log(1-hypothesis)) # 0일때 부분


train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


# tf.case(x, dtype=) : x 값을 dypte으로 리턴시킨다. bool 값일경우 0과 1값으로 리턴시킨다.
#예측값 d예측값 0.5보다 크면 1 , 작으면 0
predicted = tf.cast(hypothesis > 0.5, dtype= tf.float32)
# 예측과 값이 같으면 1 아니면 0 을 다더해서 평균
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(10001):
        c_, _ = sess.run([cost, train], feed_dict={X:x_data,Y:y_data})
        if i%200 == 0:
            print(i, c_)
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data,Y:y_data})
    print('hypothesis:', h, ' 예측값:',p, ' 정확도:', a)





