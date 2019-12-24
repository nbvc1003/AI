import tensorflow as tf

# 2.0 신버전에서 사용법
# tf.compat.v1.disable_eager_execution()

x = tf.Variable([[2,2,2], [2,2,2]]) # 2행 3열
y = tf.Variable([[3,3,3], [3,3,3]]) # 2행 3열 -> 3행 2열

# 행열의 곱은 첫번째 열의 갯수와 두번째 행열의 행의 갯수가 같다야 한다.
# transpose 로 행과 열을 바꿔서 곱을 해야 함
z = tf.matmul(x, tf.transpose(y))
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

tf.print(z)
print(z)
# print(sess.run(z))

# sess.close()