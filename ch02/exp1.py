import tensorflow as tf


x = tf.Variable([[1.,2.,3.],[4.,5.,6.]])
# exp : 자연대수 e**1, e**2..... e**5, e**6
z = tf.exp(x)
tf.print(z)

##
