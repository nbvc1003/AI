import tensorflow as tf

tf.print(tf.one_hot([[0],[1],[2],[0]], depth=3))

t = tf.one_hot([[0],[1],[2],[0]], depth=3)
tf.print(tf.reshape(t, shape=[-1,3]))

