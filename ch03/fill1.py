import tensorflow as tf

# 초기값을 1, 0 , 등등 으로 채운다. 
one1 = tf.ones([2,3], dtype=tf.float32)
zero1 = tf.zeros([2,2]) # 배열선언과 비슷

fill1 = tf.fill([3,3],9) # (3 x 3) 을 9로채운다.
tf.print(one1)
tf.print(zero1)
tf.print(fill1)



