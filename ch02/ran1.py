import tensorflow as tf

# 평균0, 표준편차 0.35 , 인 784 행, 200열 랜덤값 테이블
x = tf.Variable(tf.compat.v1.random.normal([784,200], stddev=0.35))

y = tf.Variable(x + 3)
tf.print(y)
tf.print(y.get_shape())


