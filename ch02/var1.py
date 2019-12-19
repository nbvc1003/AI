import tensorflow as tf

tf.compat.v1.disable_eager_execution()

x = tf.Variable(2.)
print(x)
sess = tf.compat.v1.Session()

# 변수 초기화 처음 항상 실행 
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run(x))
sess.close()