import tensorflow as tf
y = [[0],[1],[2],[1],[2]]

one_hot1 = tf.one_hot(y, depth=3) # 3,4,5
tf.print(one_hot1)
one_hot2 = tf.one_hot(y, depth=4) # 3,4,5
tf.print(one_hot2)
one_hot3 = tf.one_hot(y, depth=5) # 3,4,5
tf.print(one_hot3)



