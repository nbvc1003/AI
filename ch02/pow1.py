import tensorflow as tf

# 제곱
x = tf.Variable([[2,3,4],[5,6,7]])
z = tf.Variable(x,3)
tf.print(z)