import tensorflow as tf

## 구버전
# tf.compat.v1.disable_eager_execution()
# x = tf.Variable(2.)
# y = tf.Variable(x.initialized_value() + 3.0)
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
# print(sess.run(y))
# sess.close()

# 새버전
x = tf.Variable(2.0)
y = tf.Variable(x + 3.0)
tf.print(y)