import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()

x = [1.,2.,3.]
y = [1.,2.,3.]
w = tf.compat.v1.placeholder(tf.float32)
hypothesis = tf.multiply(x, w) # b를 0으로 가정
cost = tf.reduce_mean(tf.square(hypothesis-y))

sess = tf.compat.v1.Session()

sess.run(tf.compat.v1.global_variables_initializer())

w_val = []
cost_val = []
for i in range(-30, 50):
    feed_w = i * 0.1
    # w가 -3에서 5로 변할때 cost의 값 변화
    cost_, w_ = sess.run([cost, w], feed_dict={w:feed_w})
    w_val.append(w_)
    cost_val.append(cost_)


plt.plot(w_val, cost_val)
plt.show()
