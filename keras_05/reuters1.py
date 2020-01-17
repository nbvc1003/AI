from keras.datasets import reuters
import numpy as np
(train_data, train_labels ),(test_data, test_labels) = reuters.load_data(num_words=10000)

print("훈련 데이터 갯수 :",len(train_data)," 테스트 데이터 갯수:", len(test_data))
# 단어의 해당하는 인덱스
print(train_data[10])

# 숫자를 one_hot으로 변경 한다.
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

X_train = vectorize_sequences(train_data)
X_test = vectorize_sequences(test_data)


from keras.utils.np_utils import to_categorical
# label을 one_hot으로
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
# 입력데이터  10000 * 단어갯수  : 하나의 기사당
model.add(Dense(128, activation='relu', input_shape=(10000,)))
model.add(Dense(128, activation='relu'))
# 최종결과 46개로 분류
model.add(Dense(46, activation='softmax'))
#검증 데이터 셋 구성

X_val = X_train[:1000]
partial_X_train = X_train[1000:]

Y_val = one_hot_test_labels[:1000]
partial_Y_train = one_hot_train_labels[1000:]

# 모델설정
# loss : binary_crossentropy : 2개, categorical_crossentropy : 3개 이상, mse(mean_square_error) : 연속형
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(partial_X_train, partial_Y_train, epochs=8, batch_size=512, validation_data=(X_val, Y_val))
acc = model.evaluate(X_test, one_hot_test_labels)
print('정확도 :' , acc[1])

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss,'b', label="Validation Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo',label='Training acc')
plt.plot(epochs, val_acc, 'b',label='Validation acc')
plt.legend()
plt.show()




