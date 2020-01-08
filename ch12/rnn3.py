import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.InteractiveSession()
h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]

with tf.compat.v1.variable_scope('3-batch') as scope:
    x_data = np.array([[h,e,l,l,o],[e,o,l,l,l],[l,l,e,e,l]], dtype=np.float32)
    print(x_data)
    hidden_size = 2
    cell = tf.keras.layers.SimpleRNNCell(units=hidden_size)
    outputs, _states = tf.compat.v1.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    sess.run(tf.compat.v1.global_variables_initializer())
    print(outputs.eval())
    print(_states.eval())
