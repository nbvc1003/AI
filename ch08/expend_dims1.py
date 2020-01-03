import tensorflow as tf
import numpy as np

t = np.array([[[0,1,2], [3,4,5]],
              [[6,7,8], [9,10,11]]])
tf.print(t.shape)
tf.print(tf.reshape(t, shape=[-1,3]))
tf.print(tf.reshape(t, shape=[-1,1,3]))

tf.print(tf.squeeze([[0],[1],[2]])) # 2차원 -> 1차원
tf.print(tf.expand_dims([0,1,2], 1)) # 1차춴 -> 1차원 증가


