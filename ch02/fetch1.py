import tensorflow as tf

## 구버전
with tf.compat.v1.Session() as sess:
    i1 = tf.constant([3.0])
    i2 = tf.constant([4.0])
    i3 = tf.constant([5.0])
    m1 = tf.add(i2, i3)
    m2 = tf.multiply(i1, m1)
    print(sess.run([m1,m2]))

# 신버전
# i1 = tf.constant([3.0])
# i2 = tf.constant([4.0])
# i3 = tf.constant([5.0])
# m1 = tf.add(i2, i3)
# m2 = tf.multiply(i1, m1)
#
# # tf.print([m1, m2])
# tf.print([m2, m1]) # 동시에 두가지 이상 연산 실행..


