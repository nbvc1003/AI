from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax # one_hot 에서 가장큰 컬럼값
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784).astype('float')/255.0
X_test = X_test.reshape(10000, 784).astype('float')/255.0
# one_hot
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
# 훈련셋 30%를 검증
X_val = X_train[42000:]
X_train = X_train[:42000]

Y_val = Y_train[42000:]
Y_train = Y_train[:42000]

model = Sequential()
model.add(Dense(units=512, input_dim=784, activation='relu'))
model.add(Dense(units=512,  activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train,Y_train, epochs=5, batch_size=32,
          validation_data=(X_val, Y_val))

# 모델 평가
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

print('손실과 정확도 :', loss_and_metrics)

# 모델 사용
xhat_idx = np.random.choice(X_test.shape[0], 5)# 전체갯수중에 5개
xhat = X_test[xhat_idx]

# xhat 랜덤에 해당하는 y갑을 예측한값
yhat = model.predict_classes(xhat)

for i in range(5):
    print('True :', argmax(Y_test[xhat_idx[i]]), ' Pre :', yhat[i])








