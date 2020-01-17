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


# 모델 만들기
model = Sequential()
model.add(Dense(units=2, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='relu'))
# 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# 학습
hist = model.fit(X_train, Y_train, epochs=500, batch_size=10, validation_data=(X_val, Y_val))
# 학습과정 시각화
import matplotlib.pyplot as plt
fig, loss_ax = plt.subplots() # 그래프 여러개
acc_ax = loss_ax.twinx() # loss_ax와 x축을 공유

# x는 epochs수 (훈련횟수) y는 손실 훈련손실 , 노랑
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
# val 데이터 손실
loss_ax.plot(hist.history['val_loss'], 'r',label='val loss')
# x축 epoch, y는 정확도
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('lose')
# acc_ax는 loss_ax와 x축을 공유한다.
acc_ax.set_ylabel('accuracy')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()




