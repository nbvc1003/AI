from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
np.random.seed(123)

# 데이터 셋
# 전체 60000
(X_train, Y_train), (X_test,Y_test) = mnist.load_data()
# 훈련셋 50000  검증셋 10000
X_val = X_train[50000:]
Y_val = Y_train[50000:]
X_train = X_train[:50000]
Y_train = Y_train[:50000]

# X 데이터는 실수로 변경
X_train = X_train.reshape(50000,784).astype('float32')/255.0
X_val = X_val.reshape(10000,784).astype('float32')/255.0
X_test = X_test.reshape(10000,784).astype('float32')/255.0

# Y 는 one_hot 으로
Y_train = np_utils.to_categorical(Y_train)
Y_val = np_utils.to_categorical(Y_val)
Y_test = np_utils.to_categorical(Y_test)

# 모델 만들기
model = Sequential()
#       input_dim=28*28 == input_shape=[28*28,] -> 같은 의미
model.add(Dense(units=2,input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
#모델 엮기 , 구속
# categorical_crossentropy :3가지 이상 결과를 구할때
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#   validation_data= 중간에 평가 데이터로 평가를 거져 결과를 반영한다.
hist = model.fit(X_train, Y_train, epochs=10, batch_size=10, validation_data=(X_val, Y_val))
print(hist.history['loss'])
print(hist.history['accuracy'])
print(hist.history['val_loss'])
print(hist.history['val_accuracy'])