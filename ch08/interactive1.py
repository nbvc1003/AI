import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# session을 실행 시키는 다른 방법
sess = tf.compat.v1.InteractiveSession()

x = [[0,1,2],[2,1,0]]
# InteractiveSession을 사용시 실행은 .eval()을 사용한다. 
print(tf.ones_like(x).eval())
print(tf.zeros_like(x).eval())
sess.close()