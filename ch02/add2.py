import tensorflow as tf


tf.print(3. + 4.5)
tf.print(tf.add([1,3], [2,4]))

# 함수선언
@tf.function # 함수를 그래프로 그려서 작동
def add_and_triple(x,y):
    return tf.add(x,y) * 3

tf.print(add_and_triple(3.0, 4.5))