import tensorflow as tf

with tf.compat.v1.Session() as sess:
    i1 = tf.compat.v1.placeholder(tf.float32)
    i2 = tf.compat.v1.placeholder(tf.float32)
    out = i1 * i2
    # feed_dict : 실행할때 데이터를 전달하는 역활
    print(sess.run([out], feed_dict={i1:[.7],i2:[.3]}))


