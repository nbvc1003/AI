from keras.utils import np_utils
from keras.datasets import mnist
import numpy as np
from numpy import argmax
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_test = X_test.reshape(10000, 784).astype('float32')/255.0
Y_test = np_utils.to_categorical(Y_test)

# 랜덤하게 5개 가져오는 과정
xhat_idx = np.random.choice(X_test.shape[0], 5)
xhat = X_test[xhat_idx]

# 모델 불러오기
from keras.models import load_model

model = load_model('mnist_mlp.h5')
# 모델 사용 predict_classes 분류 예측
yhat = model.predict_classes(xhat)

for i in range(5):
    # argmax onehot 에서 값이 가장큰 1의 인덱스
    print('True :', argmax(Y_test[xhat_idx[i]]), ' Pred :', yhat[i])

