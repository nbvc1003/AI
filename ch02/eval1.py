import tensorflow as tf

tf.compat.v1.disable_eager_execution()

sess = tf.compat.v1.InteractiveSession()
x = tf.constant(2.0)
y = tf.constant(3.0)

z = tf.add(x,y)

# InteractiveSession 은 ecal로 실행
print(z.eval())
sess.close()