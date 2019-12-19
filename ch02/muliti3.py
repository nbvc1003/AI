import tensorflow as tf

X = tf.Variable([[2,2,2],[2,2,2]])
Y = tf.Variable([[3,3,3],[3,3,3]])

z = tf.multiply(X,Y)
tf.print(z)