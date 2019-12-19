import tensorflow as tf

## 구버전
# tf.compat.v1.disable_eager_execution()
# x = tf.compat.v1.placeholder(tf.float32)
# y = tf.compat.v1.placeholder(tf.float32)
# z = tf.multiply(x,y)
# sess = tf.compat.v1.Session()
# print(sess.run(z, feed_dict={x:3., y:5.}))
# sess.close()

## 2.0버전

x = 3.0
y = 5.0
z = tf.multiply(x,y)
tf.print(z)

