import tensorflow as tf

tf.compat.v1.disable_eager_execution()

X = tf.Variable([[2,2,2],[2,2,2]])
Y = tf.Variable([[3,3,3],[3,3,3]])
z = tf.multiply(X,Y)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run(z))
sess.close()
