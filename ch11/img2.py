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

# filter (2, 2, 1, 1) (row, col, channel, 필터갯수)
weight = tf.constant([[[[1.]],[[1.]]],[[[1]],[[1.]]]])
print('weight shape ', weight.shape)

# padding='VALID' 패딩없이 원본그대로 'SAME' 원본사이즈와 같게
# strides=[1,1,1,1] (batch(갯수), 가로, 세로, 깊이)
conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='VALID') #

# tensor 데이터를 np.array 형으로 변환 역활만..?
conv2d_img = np.swapaxes(conv2d, 0, 3) # 4차춴과 1차원을 교환?

print(conv2d)
print(conv2d_img)

for i , one_img in enumerate(conv2d_img):
    print(one_img.reshape(2,2))
    plt.subplot(1, 2, i + 1)
    # plt.imshow(one_img.reshape(2,2), cmap='gray') # 숫자가 크면 휜색
    plt.imshow(one_img.reshape(2, 2), cmap='Greys') # 숫자가 크면 검은색
plt.show()


