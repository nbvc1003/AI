import tensorflow as tf

n1 = tf.constant(3.0, tf.float32)
n2 = tf.constant(4.0)
n3 = n1 + n2

print('n1 :' , n1)
print('n3 :' , n3)
tf.print(n3)