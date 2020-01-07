import numpy as np
import tensorflow as tf
img = np.array([[[[4],[3]],
                 [[2],[1]]
                 ]],  dtype= np.float32)

# ksize ( kernerl size) [one image, width, height, one channel] : weight 대신 정의
# ksize의 갯수,채널 strides 의 갯수채널은 같게 한다.
pool = tf.nn.max_pool(img,ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')

print(pool.shape)
tf.print(pool)