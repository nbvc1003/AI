import tensorflow as tf
a = tf.constant(1.)
b = tf.constant(2.)
c = tf.constant(3., dtype=tf.float32)
d = a + b
e = tf.add(a, b)
tf.print('a =',a)
tf.print('b =',b)
tf.print('c =',c)
tf.print('d =',d)
tf.print('e =',e)
a = tf.constant([1,2,4])
b = tf.constant([[1,2,4],[5,6,7]])
c = tf.constant([[[1,2,4]],[[5,6,7]]])
tf.print('a =',a)
tf.print('b =',b)
tf.print('c =',c)
