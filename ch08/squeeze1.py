import tensorflow as tf


# squeeze 데이터 size 가 1개 짜리 차원을 제거 한다.
# 불필요한 차원을 제거 한다.
c1 = [[[1]]]
tf.print(tf.squeeze(c1))
c2 = [[[1,2,3]]]
tf.print(tf.squeeze(c2))