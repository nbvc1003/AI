from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models, layers


(train_images, train_labels),(test_images, test_labels)=mnist.load_data()

train_images = train_images.reshape((60000, 28*28)) # 28 * 28 한글자
train_images = train_images.astype('float32')/255 # 컬러값을 0~1 사이로 조절
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float')/255 # 컬러값을 0~1 사이로 조절
#정수를 one_hot 으로 변경

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = models.Sequential() # 순차적으로 층을 세울 신경망 틀
# output 512 , 층을통과할때 relu로 채크, 784개가 하나의 글자
# relu : 경사소실문제 해결 하기위한
# Dense : 데이터 전체를 받아서 처리 한다.
model.add(layers.Dense(512, activation='relu', input_shape=[28*28,]))
model.add(layers.Dense(10, activation='softmax'))
# 'sgd' : 경사하강법, 'rmsprop' :?     'categorical_crossentropy' :손실계상법?
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# 훈련데이터를 128개씩 묶어서 한번에 총 5회 (60000*5)
model.fit(train_images,train_labels, epochs=5, batch_size=128)

# 테스트 데이터 검증
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('정확도 ', test_acc)











