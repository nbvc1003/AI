import tensorflow as tf
from tensorflow_core.examples import input_data

import matplotlib.pyplot as plt
import numpy as np
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
img = mnist.train.images[0].reshape(28,28)

# cnn input 데이터형식으로 변환
img = img.reshape(-1, 28, 28,1) # -1 (알아서), 28, 28 (이미지사이즈), 1(컬러수)
#  [rol, col, 컬러, 필터수]
W1 = tf.Variable(tf.random.normal([3,3,1,5], stddev=0.01))

cond2v = tf.nn.conv2d(img, W1, strides=[1,2,2,1], padding='SAME') # 상하 2칸씩 이동 -> 14 * 14 로 줄어든다.
tf.print(cond2v.shape)
cond2v_img = np.swapaxes(cond2v, 0, 3)
for i , one_img in enumerate(cond2v_img):
    plt.subplot(1, 5, i+1)
    plt.imshow(one_img.reshape(14,14), cmap="gray") # W1값에 따라서 다양하게 표현됨
plt.show()