import tensorflow as tf
import os

tf.print(tf.__version__)
tf.print("hello")

# constant 상수값 지정
x = tf.constant(3.0)
# tf.print(x.dtype)

y = tf.constant(4.0)
z = tf.multiply(x,y)
tf.print(z, z.dtype)





