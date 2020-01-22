import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import np_utils


# 크드-> mp3 유틸
# http://lilypond.org/

np.random.seed(123)

# 손실값 출력 (그래프) 목적
class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.losses = []
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))

# 음표와 인덱스
code2idx = {'c4':0, 'd4':1, 'e4':2, 'f4':3, 'g4':4, 'a4':5, 'b4':6,
 'c8':7, 'd8':8, 'e8':9, 'f8':10, 'g8':11, 'a8':12, 'b8':13}
idx2code = {0:'c4', 1:'d4', 2:'e4', 3:'f4', 4:'g4', 5:'a4', 6:'b4',
 7:'c8', 8:'d8', 9:'e8', 10:'f8', 11:'g8', 12:'a8', 13:'b8'}

# window_size + 1 크기의 포커스 가 전체 데이터위를 움직이면서 window_size + 1개씩 추출해서 추출된 단위로 배열을 만들어 준다.
def seq2dataset(seq, window_size= 4):
    dataset_X = []
    dataset_Y = []
    for i in range(len(seq)- window_size):
        subset = seq[i:(i+window_size+1)]
        for si in range(len(subset) - 1):
            features = code2features(subset[si])
            dataset_X.append(features)
        dataset_Y.append([code2idx[subset[window_size]]])

    return np.array(dataset_X), np.array(dataset_Y)

# 코드와 박자를 분리
def code2features(code):
    features = []
    features.append(code2scale[code[0]]/float(max_scale_value))
    features.append(code2length[code[1]])
    return features


# 테이터 준비
code2scale = {'c':0,'d':1,'e':2,'f':3,'g':4,'a':5,'b':6}
code2length = {'4':0,'8':1}
max_scale_value = 6.0


# 나비야 노래 악보를 코드값으로 음계+길이
seq = ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'd8', 'e8', 'f8', 'g8', 'g8', 'g4',
 'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4',
 'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4',
 'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4']


# 2. 데이터셋 생성하기
X_train, y_train = seq2dataset(seq, window_size = 4)
# 입력을 (샘플 수, 타임스텝, 특성 수)로 형태 변환
X_train = np.reshape(X_train, (50, 4, 2))
# 라벨값에 대한 one-hot 인코딩 수행
y_train = np_utils.to_categorical(y_train)
one_hot_vec_size = y_train.shape[1]

print("one hot encoding vector size is ", one_hot_vec_size)

# 모델 구성
model = Sequential()
model.add(LSTM(128, batch_input_shape=(1,4,2), stateful=True))
model.add(Dense(one_hot_vec_size, activation='softmax'))

#모델설정
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )
num_epochs = 2000
history = LossHistory()
for epoch_idx in range(num_epochs):
    print("epoch_idx 번 :", epoch_idx)
    model.fit(X_train, y_train, epochs=1, batch_size=1, shuffle=False, callbacks=[history])
    model.reset_states()


# 그래프
import matplotlib.pyplot as plt
plt.plot(history.losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'],loc='upper left')
plt.show()
# 모델 평가
score = model.evaluate(X_train, y_train, batch_size=1)
print('정확도 : {0:.2f}'.format(score[1]))
model.reset_states()


pred_count = 50 # 최대 예측 개수 정의
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 훈련데이터로 예측
seq_out = ['g8','e8','e4','f8']
pred_out = model.predict(X_train, batch_size=1) # X_train 의 갯수가 여러 개 이기 때문에 batch_size를 지정해야한다.
for i in range(pred_count):
    idx = np.argmax(pred_out[i])# 예측한 데이터의 인덱스 번호
    seq_out.append(idx2code[idx]) # 인덱스 숫자를 코드로 변경
print('one :', seq_out)

model.reset_states()
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 예측데이터를 베이스 데이터에 포함해서 전곡 예측
seq_in = ['g8', 'e8', 'e4', 'f8']
seq_out = seq_in

seq_in_featrues = []

for si in seq_in:
    features = code2features(si)
    seq_in_featrues.append(features)
for i in range(pred_count):
    sample_in = np.array(seq_in_featrues)
    sample_in = np.reshape(sample_in, (1, 4, 2))  # 샘플 수, 타입스텝 수, 속성 수
    pred_out = model.predict(sample_in)
    idx = np.argmax(pred_out)
    seq_out.append(idx2code[idx])

    features = code2features(idx2code[idx])
    seq_in_featrues.append(features)
    seq_in_featrues.pop(0)
print('full ', seq_out)














