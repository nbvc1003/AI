import tensorflow as tf

x = [1,4]
y = [2,5]
z = [3,6]

# 배열을 쌓는다.
# default axit=0 (열 방향)
tf.print(tf.stack([x,y,z]))

# 행 방향으로 쌓는다.
tf.print(tf.stack([x,y,z], axis=1))