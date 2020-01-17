from keras import models, layers
from keras.datasets import imdb
import numpy as np

# imdb : 영화평가 글 db 5만건
# 영화평가글 중에서 자주 사용하는 단어 10000개까지만
(Xdata_train, Ydata_train), (Xdata_test, Ydata_test) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


# print(X_train.shape)
# print(X_train[0])
X_train = vectorize_sequences(Xdata_train)
X_test = vectorize_sequences(Xdata_test)
# print(X_train.shape)
# print(X_train[0])

Y_train = np.asanyarray(Ydata_train).astype('float32')
Y_test = np.asanyarray(Ydata_test).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
X_val = X_train[:10000]
partial_X_train = X_train[10000:]
Y_val = Y_train[:10000]
partial_Y_train = Y_train[10000:]
history = model.fit(partial_X_train, partial_Y_train, epochs=4, batch_size=512, validation_data=(X_val, Y_val))

results = model.evaluate(X_test, Y_test)
print('loss and accuracy :', results)

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epoch = range(1, len(loss)+1)
# 파란색 점
plt.plot(epoch, loss, 'bo', label='Train loss')
plt.plot(epoch, val_loss, 'b', label='val loss')
plt.title("Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epoch, acc, 'ro', label='Train accuracy')
plt.plot(epoch, val_acc, 'r',label='Val accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


