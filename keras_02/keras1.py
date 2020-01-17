from keras.datasets import mnist
import numpy as np

# train 60000 (28 * 28 사이즈)  0 ~ 255 사이의 숫자
(train_images, train_labels),(test_images, test_labels)=mnist.load_data()
# 글자 28*28 = 784 합해야 글자 하나 , 실수 0 ~ 1 사이의 값으로 변경
X_train = train_images.reshape(60000, 28*28).astype(np.float)/255.0
# print(X_train.shape)
X_test = test_images.reshape(10000, 28*28).astype(np.float)/255.0


