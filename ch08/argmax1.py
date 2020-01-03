import tensorflow as tf

x = [[0,1,2],
     [2,1,0]]

# 각 기준별 가장큰 값
tf.print(tf.argmax(x, axis=0)) # 열
tf.print(tf.argmax(x, axis=1)) # 행
tf.print(tf.argmax(x, axis=-1)) # 행과 같다. ?
