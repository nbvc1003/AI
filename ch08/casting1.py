import tensorflow as tf

# 형변환
tf.print(tf.cast([1.8,2.2,3.3,4.9], tf.int32))

# True = 1, False = 0 으로 형변환
tf.print(tf.cast([True, False, 1==1, 0==1], tf.int32)) 
