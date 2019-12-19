import tensorflow as tf

tf.compat.v1.disable_eager_execution()
x = tf.Variable([[2.,2.,2.],[2.,2.,2.]])
z = tf.compat.v1.log(x)

sess = tf.compat.v1.Session()
sess.run((tf.compat.v1.global_variables_initializer()))
print(sess.run(z))
sess.close()

