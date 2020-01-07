import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
image = np.array([[
                [[1],[2],[3]],
                [[4],[5],[6]],
                [[7],[8],[9]]
                    ]], dtype=np.float32)
# 이미지는 (갯수, row, col, channel)
print("image shape ", image.shape)

# filter (2, 2, 1, 3) (row, col, channel, 필터갯수)
weight = tf.constant([[[[1.,10.,-1.]],[[1.,10.,-1.]]],[[[1.,10.,-1.]],[[1.,10.,-1.]]]])
# 원본그대로, 10배, -1배

print('weight.shpage :', weight.shape)
# strides=[1,1,1,1] (batch(갯수), 가로, 세로, 깊이(컬러))
conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='SAME')
conv2d_img = np.swapaxes(conv2d, 0, 3)

print(conv2d)
print(conv2d_img)

for i , one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,3, i+1)
    plt.imshow(one_img.reshape(3,3), cmap='gray')
plt.show()

