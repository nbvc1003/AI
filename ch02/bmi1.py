import tensorflow as tf
tf.compat.v1.disable_eager_execution()
X = tf.compat.v1.placeholder(tf.float32)
Y = tf.compat.v1.placeholder(tf.float32)
w = tf.Variable(tf.compat.v1.random_normal([1]),tf.float32)
# BMI 단위 키는 m 몸무게 kg
hypothesis = w * X * X
loss = tf.reduce_mean(tf.square(hypothesis-Y))
train = tf.compat.v1.train.GradientDescentOptimizer(
    learning_rate=0.01).minimize(loss)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(1001):
    t_,w_,l_,h_=sess.run([train,w,loss,hypothesis],
         feed_dict={X:[1.6,1.7,1.8], Y:[55,60,65]})
    if i%20 == 0:
        print(i, w_, l_)
# 위에서 구한 w(마지막 w)를 대입하여 예측
h = sess.run(hypothesis, feed_dict={X:[1.5,1.65,1.9]})
print("몸무게 예측 :",h)
sess.close()