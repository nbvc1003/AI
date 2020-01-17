import numpy as np
from keras.models import Sequential
from keras.layers import Dense
np.random.seed(123)
dataset = np.loadtxt('pima-indians-diabetes.data', delimiter=',')
#데이터 셋
X_train = dataset[:700, :8]
Y_train = dataset[:700, 8]
X_test = dataset[700: , :8]
Y_test = dataset[700: , 8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8,  activation='relu'))
model.add(Dense(1,  activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=1500, batch_size=64)
scores = model.evaluate(X_test, Y_test)
print("정확도 : ", scores[1])
