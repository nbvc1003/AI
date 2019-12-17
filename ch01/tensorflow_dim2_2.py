import tensorflow as tf

mat1 = tf.constant([[3.0,3.0]])
mat2 = tf.constant([[2.0],[2.0]])

# 곱하기
# 열과 행이 맞지 않을 경우 강제로 한쪽에 맞게 0으로 채워진 행열로 맞춰서 계산
# 브로드 케스트 라고 한다.
mul = tf.multiply(mat1, mat2)
tf.print(mul, mul.dtype)