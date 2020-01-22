import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np

# 음표와 인덱스
code2idx = {'c4':0, 'd4':1, 'e4':2, 'f4':3, 'g4':4, 'a4':5, 'b4':6,
 'c8':7, 'd8':8, 'e8':9, 'f8':10, 'g8':11, 'a8':12, 'b8':13}
idx2code = {0:'c4', 1:'d4', 2:'e4', 3:'f4', 4:'g4', 5:'a4', 6:'b4',
 7:'c8', 8:'d8', 9:'e8', 10:'f8', 11:'g8', 12:'a8', 13:'b8'}



# window_size + 1 크기의 포커스 가 전체 데이터위를 움직이면서 window_size + 1개씩 추출해서 추출된 단위로 배열을 만들어 준다.
def seq2dataset(seq, window_size):
    dataset = []
    for i in range(len(seq)- window_size):
        subset = seq[i:(i+window_size+1)]
        dataset.append([code2idx[item] for item in subset])
    return np.array(dataset)

# 나비야 노래 악보를 코드값으로 음계+길이
seq = ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'd8', 'e8', 'f8', 'g8', 'g8', 'g4',
 'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4',
 'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4',
 'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4']

np.random.seed(123)

# 손실값 출력 (그래프) 목적
class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.losses = []
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))


dataset = seq2dataset(seq, window_size=4)
print(dataset.shape)
# print(dataset)

# X, y 분리
X_train = dataset[:,:4] # 행에서 4 번째 열까지
y_train = dataset[:,4] # 행에서 5번째 열만
max_index_value = 13 # 데이터가 14건 위에서 정의한 전체 음표 종류

# 입력값 정규화 (최대값 나누면 데이터 0~1 사이로 변경)
X_train = X_train /float(max_index_value) #

#라밸값을 one_hot 으로 변경
y_train = np_utils.to_categorical(y_train)
one_hot_vec_size = y_train.shape[1]
print(one_hot_vec_size)

# 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=4, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(one_hot_vec_size, activation='softmax'))

# 학습과정 설정
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = LossHistory()
# 모델 학습
model.fit(X_train, y_train, epochs=400, batch_size=10, callbacks=[history])



# 학습과정 그래프 출력
import matplotlib.pyplot as plt
plt.plot(history.losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

# 모델 평가
scores = model.evaluate(X_train,y_train)
print('정확도 :', scores[1]*100, '%')
# 모델 사용
pred_count = 50
# 처음 한스탭은 어쩔수 없이 입력해줌
seq_out = ['g8','e8','e4','f8']
pred_out = model.predict(X_train)
for i in range(pred_count):
    idx = np.argmax(pred_out[i]) # one_hot -> 인덱스 값으로
    seq_out.append(idx2code[idx]) # 인덱스를 코드로 변경
print('seq_out 예측', seq_out)




# 한소설로 전곡 예측
# 예측해서 나온 결과 값을 다시 예측 베이스 데이터로 사용하는 방식
seq_in = ['g8','e8','e4','f8']
seq_out = seq_in

# max_index_value 로 나누면 값이 0~1 이므로 크기가 비슷해진다.
seq_in = [code2idx[it] / float(max_index_value) for it in seq_in]
for i in range(pred_count):
    sample_in = np.array(seq_in)
    sample_in = np.reshape(sample_in, (1,4)) # (batch_size, feature)
    pred_out = model.predict(sample_in)
    idx = np.argmax(pred_out)
    seq_out.append(idx2code[idx])
    seq_in.append(idx/float(max_index_value))
    seq_in.pop(0) # 가장앞데이터를 삭제
print('full pred :', seq_out)






