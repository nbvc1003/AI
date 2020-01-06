import tensorflow as tf
# Variable로 변수를 생성하면 메모리를 사용
tf.compat.v1.disable_eager_execution()

def func1(x):
    w = tf.Variable(2, name='w')
    b = tf.Variable(3, name='b')
    return x*w+b

print(tf.compat.v1.global_variables()) # 메모리 사용
print("===========================================")
# reuse=tf.compat.v1.AUTO_REUSE : 메모리를 사용하지 않으면 재사용
with tf.compat.v1.variable_scope('scope1', reuse=tf.compat.v1.AUTO_REUSE):
    y1 = func1(1)
    y2 = func1(2)
    print(tf.compat.v1.global_variables())
    print("===========================================")

with tf.compat.v1.variable_scope('scope2', reuse=tf.compat.v1.AUTO_REUSE):
    y1 = func1(1)
    y2 = func1(2)
    print(tf.compat.v1.global_variables())