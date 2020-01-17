
from keras import backend as K

K.tensorflow_backend._get_available_gpus()


#  특정 GPU 선택
with K.tf.device('/gpu:0'):
    model = Sequential()
    model.add(Dense(512, input_dim=28*28, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    h = model.fit(X_train, y_train_cat, batch_size=128*4, epochs=10, verbose=1, validation_split=0.3)