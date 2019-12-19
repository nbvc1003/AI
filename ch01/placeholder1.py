import tensorflow as tf

# 구버전의 소스를 사용하기 위한 사전 명령라인
tf.compat.v1.disable_eager_execution()

# 1차원 배열 연산
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

z = tf.compat.v1.multiply(x,y)

sess = tf.compat.v1.Session()
print('1  :', sess.run(z, feed_dict={x: [3.,3.], y: [5.,5.]}))
sess.close()

# 2차원 배열 연산
x = tf.compat.v1.placeholder(tf.float32, shape=(2,2))
y = tf.compat.v1.placeholder(tf.float32, shape=(2,2))
z = tf.multiply(x, y) # 같은 위치끼리 단순 곱샘
sess = tf.compat.v1.Session()
print('2  :', sess.run(z, feed_dict={x: [[3.,3.],[3.,3.]], y: [[5.,5.],[5.,5.]]}))
sess.close()
