import tensorflow as tf


t = tf.constant([1,2,3,4])
tf.print(tf.shape(t))

t = tf.constant([[1,2],[3,4]])
tf.print(tf.shape(t))
t = tf.constant([
                    [[1,2,3,4],
                     [5,6,7,8],
                     [9,10,11,12]],

                    [[13,14,15,16],
                     [17,18,19,20],
                     [21,22,23,24]]
                ])
tf.print(tf.shape(t)) # 바깥쪽부터 안쪽으로  2, 3, 4