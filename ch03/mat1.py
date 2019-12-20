import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

X = [1, 2, 3]
Y = [1, 2, 3]

w = tf.compat.v1.placeholder(tf.float32)
hypotheses = X * w
cost = tf.reduce_mean(tf.square(hypotheses - Y))
sess = tf.compat.v1.Session()

sess.run(tf.compat.v1.global_variables_initializer())
w_val = []
cost_val = []
for i in range(-30, 50):
    feed_W = i * 0.1
    cur_c, cur_w = sess.run([cost,w], feed_dict={w:feed_W})
    w_val.append(cur_w)
    cost_val.append(cur_c)

plt.plot(w_val, cost_val)
plt.show()
sess.close()