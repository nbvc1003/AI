import tensorflow as tf

# 평균0 표준편차1 인데이터 3건  : 0에 가까울수록 많다.
tf.print(tf.random.normal([3]))
tf.print(tf.random.normal([2]))
tf.print(tf.random.normal([2, 3]))



