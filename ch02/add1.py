import tensorflow as tf


tf.compat.v1.disable_eager_execution()
# 변수타입정의
a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)
# 새션생성
sess = tf.compat.v1.Session()
# 기능정의
add1 = a + b

# 동작과 값을 넣고 동작
print(sess.run(add1, feed_dict={a:3, b:4.5}))

print(sess.run(add1, feed_dict={a:[1,3], b:[2,4]}))
triple = add1 * 3.0
print(sess.run(triple, feed_dict={a:3,b:4.5}))
sess.close()