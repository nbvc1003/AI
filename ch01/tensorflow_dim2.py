import tensorflow as tf

mat1 = tf.constant([[3.1,3.2]])
mat2 = tf.constant([[2.1,2.2]])

mul = tf.multiply(mat1, mat2)
tf.print(mul, mul.dtype)