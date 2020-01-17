import keras
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

# 데이터 고르기 random 하게 선택
train_rand_idxs = np.random.choice(50000,700) # 숫자 0~10000 에서 700개 랜덤한 숫자 선택
val_rand_idxs = np.random.choice(10000,300)

X_train = X_train[train_rand_idxs]
Y_train = Y_train[train_rand_idxs]
X_val = X_val[val_rand_idxs]
Y_val = Y_val[val_rand_idxs]
X_test = X_test[val_rand_idxs]
Y_test = Y_test[val_rand_idxs]



# 모댈 생성
model = Sequential()
model.add(Dense(units=2, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
#모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# 학습
tb_hist = keras.callbacks.TensorBoard( log_dir='.\log', histogram_freq=0, write_graph=True, write_images=True)
model.fit(X_train, Y_train, epochs=50, batch_size=10, validation_data=(X_val, Y_val), callbacks=[tb_hist])

