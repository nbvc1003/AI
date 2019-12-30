import tensorflow as tf

y = [0, 1,2,3]

# 
one_hot1 = tf.one_hot(y, 4) # one hot 의 크기 4
one_hot2 = tf.one_hot(y, 5)

tf.print(one_hot1)
print(' {}'.format(one_hot2))

