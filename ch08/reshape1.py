import tensorflow as tf

c1 = tf.constant([1,2,3,4,5,6,7,8,9,10,11,12])
tf.print(tf.reshape(c1, [3,4]))

#  4열 행은 알아서정해라.
tf.print(tf.reshape(c1, [-1,4]))

c2 = tf.constant([[1,2],[3,4]])

# -1 : 알아서 추측해서
tf.print(tf.reshape(c2, [-1]))

