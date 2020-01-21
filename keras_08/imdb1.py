from keras.datasets import imdb
from keras import preprocessing

# 사용할 단어수
max_features = 1000
# 텍스트 길이
max_len = 20

#이롬설명 참고
# https://subinium.github.io/Keras-6-1/

# 데이터 셋 생성
# 데이터 로드
# 영화평 중 자주 사용하는 단어 1000개만
(X_train, y_train),(X_test, y_test) = imdb.load_data(num_words=max_features)

# 모든 영화평을 20개 단어로 길이로 동일하게 만듦
X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)
# 모델 구축
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
model = Sequential()
model.add(Embedding(10000, 8, input_length=max_len))
#출력갯수: samples, max_length * 8
model.add(Flatten())
model.add(Dense(1, activation='sigmoid')) # 긍정/ 부정 평가
# 모델 설정
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test,y_test))
model.summary()







