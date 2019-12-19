import tensorflow as tf

## 2.0.0 이전 버전의 소스를 실행하기 위한 방법

## 구버전 소스를 사용하기 위한 방법 case 1
##=============================================================
# 구버전의 소스를 사용하기 위한 사전 명령라인
tf.compat.v1.disable_eager_execution()
## 예젼 버전 소스내용.
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
z = tf.multiply(x, y)
sess = tf.compat.v1.Session()
print(sess.run(z, feed_dict={x: 3., y: 5.}))
sess.close()
##=============================================================


## 구버전 소스를 사용하기 위한 방법 case 2
##=============================================================
# with ~ as 안쪽에 소스가 있어야 한다.
with tf.compat.v1. Session() as sess:
    x = tf.compat.v1.placeholder(tf.float32)
    y = tf.compat.v1.placeholder(tf.float32)
    z = tf.multiply(x, y)
    print(sess.run(z, feed_dict={x: 3., y: 5.}))
##=============================================================

