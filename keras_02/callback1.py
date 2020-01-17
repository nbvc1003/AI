import keras
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
# 사용자정의 콜백클래스

class Customhistory(keras.callbacks.Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

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

model = Sequential()
model.add(Dense(units=2, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])


from keras.callbacks import EarlyStopping
# patience값 만큼 더 증가후 중지..
early_stop = EarlyStopping(patience=2)

cust_hist = Customhistory()


# 초기화를 제거 하기 위하여 epochs= 1000와 같이 할경우 model.reset_states() 가 작동

model.fit(X_train, Y_train, epochs=40, batch_size=10, validation_data=(X_val, Y_val),
          callbacks=[cust_hist, early_stop])

# 모델 학습 시각화
import matplotlib.pyplot as plt
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx() # x축공유하는 plot 생성 twiny() -> y축 공유하는 plot
acc_ax.plot(cust_hist.train_loss, 'y', label='train loss')
acc_ax.plot(cust_hist.val_loss, 'r', label='val loss')
acc_ax.plot(cust_hist.train_acc, 'b', label='train acc')
acc_ax.plot(cust_hist.val_acc, 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')
loss_ax.legend(loc='upper right')
acc_ax.legend(loc='lower left')
plt.show()




