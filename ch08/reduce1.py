import tensorflow as tf

tf.print(tf.reduce_mean([1,2], axis=0)) # axis=0 열단위로 계산 1차원 배열은
x = [[1.,2.,],
     [3.,4.]]

tf.print(tf.reduce_mean(x)) # 전체 원소 평균값
tf.print(tf.reduce_mean(x, axis=0))# axis=0 열단위로 계산
tf.print(tf.reduce_mean(x, axis=1))
tf.print(tf.reduce_mean(x, axis=-1)) # -1 은 마지막 차원

tf.print(tf.reduce_sum(x)) # 1+2+3+4
tf.print(tf.reduce_sum(x, axis=0))
tf.print(tf.reduce_sum(x, axis=1))
tf.print(tf.reduce_sum(x, axis=-1)) # -1 차원이 여러개일경우 마지막 차원

tf.print(tf.reduce_sum(x, axis=1)) # 행으로 더한다.
tf.print(tf.reduce_mean(tf.reduce_sum(x, axis=1))) # 행으로 더하고 평균







