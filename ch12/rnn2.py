import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.InteractiveSession()
# hello 중복되지 않은 스팰링을 뽑아서 순서 부여
h = [1,0,0,0] # 0
e = [0,1,0,0] # 1
l = [0,0,1,0] # 2
o = [0,0,0,1] # 3

with tf.compat.v1.variable_scope('two_cell') as scope:
    hidden_size = 2
    cell = tf.keras.layers.SimpleRNNCell(units=hidden_size)
    print(cell.output_size, cell.state_size ) # 2, 2
    x_data = np.array([[h,e,l,l,o]], dtype=np.float32) #
    print(x_data)
    output, state = tf.compat.v1.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    sess.run(tf.compat.v1.global_variables_initializer())
    print(output.eval()) # .evl() 실행