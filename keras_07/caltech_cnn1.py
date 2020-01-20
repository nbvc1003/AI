from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import numpy as np

categories = ["chair", "camera","butterfly","elephant","flamingo"]
nb_classes = len(categories)
# 이미지 크리 지정
image_w = 64
image_h = 64
# 데이터 가져 오기
X_train, X_test, y_train, y_test = np.load("./images/5obj.npy", allow_pickle=True)
#데이터 정규화
X_train = X_train.astype('float')/256
X_test = X_test.astype('float')/256
print(X_train.shape, X_test.shape)

# 모델 구축
model = Sequential()
model.add(Convolution2D(32, kernel_size=(3,3), padding='same', input_shape=X_train.shape[1:])) # (250, 64, 64, 3) 배열의 뒤쪽 3자리만
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64,(3,3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64,(3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
#학습설정
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=50)
# 모델평가
scores = model.evaluate(X_test, y_test)
print('loss :', scores[0])
print('acc :', scores[1])


