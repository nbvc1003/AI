import tensorflow as tf

value = [[1,2], [3,4],[5,6],[7,8]]

s1 = tf.random.shuffle(value) # 섞어준다.
tf.print(s1)

