import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(12)
# 데이터 셋
train_datagram = ImageDataGenerator(rescale=1. / 255)  # 컬러값을 0~1 로

# flow_from_directory : 하위 디렉토리 별로 class index 지정
train_generator = train_datagram.flow_from_directory('warehouse/handwriting_shape/train',  # 위치
                                                     target_size=(24, 24),  # 이미지 사이즈
                                                     batch_size=3,
                                                     class_mode='categorical') # 2D one_hot 반환 -> 흑백으로
test_datagram = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagram.flow_from_directory('warehouse/handwriting_shape/test',  # 위치
                                                   target_size=(24, 24),  # 이미지 사이즈
                                                   batch_size=3,
                                                   class_mode='categorical')
# 모델 
model = Sequential()
# kernel_size : 컬벌루션 창의 사이즈
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(24,24,3))) # 24*24 , 3개의 필터
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 모델 설정
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
# fit_generator 제너레이트된 데이터 트레인 이미 클래스 분류가 디렉토리 별로 0,1,2 분류되어 있는 데이터
model.fit_generator(train_generator, steps_per_epoch=15, epochs=50, validation_data=test_generator, validation_steps=15)
scores = model.evaluate_generator(test_generator, steps=5)
print(scores)

# 모델 사용
output = model.predict_generator(test_generator, steps=5)

# 데이터를 소수점 3째 자리까지
# 프린트 print 함수 표현에 자동 적용
np.set_printoptions(formatter= {'float':lambda x:"{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)



