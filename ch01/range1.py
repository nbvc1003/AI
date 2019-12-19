import tensorflow as tf


tensor = tf.range(10)
tf.print(tensor)
tf.print(tf.range(1,10, 2))
tensor_a = tf.range(2)
tensor_b = tensor_a * 2 # 배열각각에 *2
tf.print(tensor_a, tensor_b)



