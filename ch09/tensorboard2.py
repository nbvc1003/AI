import tensorflow as tf

tf.compat.v1.disable_eager_execution()

X = tf.compat.v1.placeholder(tf.float32)
Y = tf.compat.v1.placeholder(tf.float32)

add = tf.add(X,Y)
mul = tf.multiply(X,Y)
add_hist = tf.compat.v1.summary.scalar('add_scalar', add)
mul_history = tf.compat.v1.summary.scalar('mul_scalar', mul)

merged = tf.compat.v1.summary.merge_all()
# merged = tf.compat.v1.summary.merge([add_hist, mul_history])# 위와 동일

with tf.compat.v1.Session() as sess:
    write = tf.compat.v1.summary.FileWriter("E:/nbvc/python/AI/ch09/log/sample_2", sess.graph)
    for i in range(100):
        summry = sess.run(merged, feed_dict={X:i*1.0,Y:2.0})
        write.add_summary(summry, i)


