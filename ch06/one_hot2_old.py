import tensorflow as tf
## for 2.0 버전
tf.compat.v1.disable_eager_execution()

y = [[0],[1],[2],[1],[2]]

one_hot1 = tf.one_hot(y, depth=3)
one_hot2 = tf.one_hot(y, depth=4)
one_hot3 = tf.one_hot(y, depth=5)


sess = tf.compat.v1.Session()
print(sess.run(one_hot1))
print(sess.run(one_hot2))
print(sess.run(one_hot3))
sess.close()