import tensorflow as tf
tf.compat.v1.disable_eager_execution()

a = tf.constant(3.0)
b = tf.constant(5.0)

c = a * b

c_summary = tf.compat.v1.summary.scalar('point',c)
merged = tf.compat.v1.summary.merge_all()

with tf.compat.v1.Session() as sess:
    write = tf.compat.v1.summary.FileWriter("E:/nbvc/python/AI/ch09/log/sample_1", sess.graph)

    result = sess.run([merged])
    write.add_summary(result[0])

# 텐서보드 실행
# tensorboard --logdir=E:/nbvc/python/AI/ch09/log --port=6006
# default 포트 6006
