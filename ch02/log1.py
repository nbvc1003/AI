import tensorflow as tf

x = tf.Variable([[2.,2.,2.],[2.,2.,2.]])

# Log  tf.log x
z = tf.compat.v1.log(x)
tf.print(z)


