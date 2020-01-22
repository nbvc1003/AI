import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import np_utils
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

# 크드-> mp3 유틸
# http://lilypond.org/


# 나비야 노래 악보를 코드값으로 음계+길이
seq = ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'd8', 'e8', 'f8', 'g8', 'g8', 'g4',
 'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4',
 'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4',
 'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4']

# window_size + 1 크기의 포커스 가 전체 데이터위를 움직이면서 window_size + 1개씩 추출해서 추출된 단위로 배열을 만들어 준다.
def seq2dataset(seq, window_size= 4):
    dataset = []
    for i in range(len(seq)- window_size):
        subset = seq[i:(i+window_size+1)]
        dataset.append([code2idx[item] for item in subset])
    return np.array(dataset)

data = seq2dataset(seq, window_size=4)
print(data.shape)

X_train = data[:,:4]
y_train = data[:,4]
max_idx_value = 13

#입력 값을 정규화
X_train = X_train/float(max_idx_value)
#LSTM 입력 (샘플수, 데이터 스탭, 특성)
X_train = np.reshape(X_train, (50, 4,1))
y_train = np_utils.to_categorical(y_train)
print('X_train :',X_train.shape)
print("y_train :",y_train.shape)
one_hot_vec_size = y_train.shape[1]

# 모델 구성
model = Sequential()
model.add(LSTM(128, batch_input_shape=(1,4,1), stateful=True))
model.add(Dense(one_hot_vec_size, activation='softmax'))

#모델설정
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )
num_epochs = 1000
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

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 훈련데이터로 예측
seq_out = ['g8','e8','e4','f8']
pred_out = model.predict(X_train, batch_size=1) # X_train 의 갯수가 여러 개 이기 때문에 batch_size를 지정해야한다.
for i in range(50):
    idx = np.argmax(pred_out[i])# 예측한 데이터의 인덱스 번호
    seq_out.append(idx2code[idx]) # 인덱스 숫자를 코드로 변경
print('one :', seq_out)

model.reset_states()
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 예측데이터를 베이스 데이터에 포함해서 전곡 예측
seq_out = ['g8','e8','e4','f8']
seq_in = seq_out
seq_in = [code2idx[it]/float(max_idx_value) for it in seq_in] # 코드-> 숫자 -> 표준화
for i in range(50):
    sample_in = np.array(seq_in)
    # (샘플수, 데이터 스탭, 특성수)
    sample_in = np.reshape(sample_in, (1,4,1))  # (batch_size, feature)
    pred_out = model.predict(sample_in)
    idx = np.argmax(pred_out)
    seq_out.append(idx2code[idx])
    seq_in.append(idx / float(max_idx_value))
    seq_in.pop(0)  # 가장앞데이터를 삭제
print('full ', seq_out)














