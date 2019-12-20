import tensorflow as tf

n1 = tf.random.normal([2,2]) # 평균0 표준편차1 인 2,2 배열 생성
n2 = tf.random.normal([2,2], mean=1.1) # 평균 1.1
n3 = tf.random.normal([2,2], mean=1.1, stddev=1.2) # 평균 1.1 표준편차 1.2
n4 = tf.random.normal([2,2], mean=1.1, stddev=1.2, seed=(123)) # 평균 1.1 표준편차 1.2 시드지정
tf.print(n1)
tf.print(n2)
tf.print(n3)
tf.print(n4)
tf.print('============================================')
# 0과 1사이의 균등한 조건의 값을 랜덤하게 추출
n2 = tf.random.uniform([2,2]) # 2,2 배열에 균등값
# normal 의 경우 가운데 값이 가장 많이 나온다.

n3 = tf.random.uniform([2,2], minval=1, maxval=5) # 최대최소 값 지정
n4 = tf.random.uniform([2,2], minval=1, maxval=5, seed=(123)) # 최대최소 값 지정

tf.print(n2)
tf.print(n3)
tf.print(n4)