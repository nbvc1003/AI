import tensorflow as tf
tf.compat.v1.disable_eager_execution()
x = tf.Variable(tf.compat.v1.random_normal([784,200],stddev=0.35))
y = tf.Variable(x.initialized_value() + 3.)
saver = tf.compat.v1.train.Saver()
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
saver.save(sess,"e:/gov/save/model.ckpt")
sess.close()