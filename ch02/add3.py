import tensorflow as tf

# 일반적으로 소문자는 데이터 1건 대문자는 데이터 여러건

X = tf.Variable([[2,2,2], [2,2,2]])
Y = tf.Variable([[3,3,3], [3,3,3]])

z = tf.add(X,Y)
tf.print(z)