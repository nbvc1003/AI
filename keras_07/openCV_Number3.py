from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
image_w = 28
image_h = 28
nb_classes = 10

def main():
    (X_train, y_train),(X_test,y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, image_w*image_h).astype('float32')/255
    X_test = X_test.reshape(10000, image_w*image_h).astype('float32')/255
    y_train = np_utils.to_categorical(y_train, 10) # one_hot으로
    y_test = np_utils.to_categorical(y_test, 10) # one_hot으로
    #모델 구축
    model = build_model()
    model.fit(X_train,y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test))

    #모델 저장
    model.save_weights('mnist.hdf5')
    model.save('mnist1.h5')
    # 모델 평가
    scores = model.evaluate(X_test,y_test)
    print(scores)

def build_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    # 모델 설정
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    main()
