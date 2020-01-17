from keras.datasets import boston_housing
# 보스톤 집값  범죄율, 세율,....
(train_data, train_targets),(test_data,test_targets)=boston_housing.load_data()
print(train_data.shape, test_data.shape)
print(test_data[0])
# 데이터 크기(0. ~ 666)가 차이가 크면 학습이 안됨
# => min_max (현재값 - 최소)/(최대 - 최소)
# => 표준화  (현재값 - 평균)/표준편차
mean = train_data.mean(axis=0) # 컬럼 평균
std = train_data.std(axis=0)   # 컬럼 표준편차
train_data = (train_data - mean) / std
# 시험데이터도 훈련데이터의 평균과 표준편차를 이용
test_data  = (test_data - mean)  / std
# 모델 구성
from keras.models import Sequential
from keras.layers import Dense
# K fold로 여러개의 모델로 테스트 할 경우
def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu',
                    input_shape=(train_data.shape[1],)))
    model.add(Dense(64,activation='relu'))
    # 회귀직선 이므로 1, 연속적 데이터
    model.add(Dense(1))
    # 모델 설정  mse연속형 데이터 mean square error
    model.compile(optimizer='rmsprop', loss='mse',metrics=['mae'])
    #-회귀는 분류에서 사용했던 것과는 다른 손실 함수를 사용
    #(= MSE는 회귀에서 자주 사용되는 손실 함수)
    #-정확도 개념은 회귀에선 일반적인 회귀 지표로, MAE 사용
    return model
# K fold
import numpy as np
k = 4
# // 나눈 몫 정수만 나옴
num_value_samples = len(train_data) // 4
num_epochs = 100 # 훈련 횟수
all_scores = []
all_mae_history = []

for i in range(k):
    print('처리중인 폴드 #', i)
    val_data=train_data[i*num_value_samples:(i+1)*num_value_samples]
    val_targets=train_targets[i*num_value_samples:(i+1)*num_value_samples]
    # 훈련 데이터 준비 : 다른 분할 전체
    # concatenate 두개의 행열을 합함
    partial_train_data=np.concatenate(
        [train_data[:i*num_value_samples],
         train_data[(i+1)*num_value_samples:]],axis=0 )
    partial_train_targets = np.concatenate(
        [train_targets[:i*num_value_samples],
         train_targets[(i+1)*num_value_samples:]], axis=0 )
    model = build_model()
    # 모델 훈련
    history = model.fit(partial_train_data,partial_train_targets,
              epochs=num_epochs,batch_size=1,verbose=0, validation_data=(val_data, val_targets))
    #예측값과 실제값차이 평균
    mae_history = history.history['mae'] # old : val_mean_absolute_error
    all_mae_history.append(mae_history)

# 4개를 평균내 500번 ( 훈련 500번 했으므로)
# average_mae_history = [np.mean([x[i] for x in all_mae_history])
#                        for i in range(num_epochs)
#                        ]

average_mae_history = [ np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)]

# print(all_scores)
# print(np.mean(all_scores))
import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history)+1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation ')
plt.show()

# 이전 데이터 90% 이번 데이터 10%반영 -> 데이터 부드럽게
def smooth_curve(points, factor= 0.9):
    smooth_points = []
    for point in points:
        if smooth_points:
            # 이전 마지막 데이터 previus
            previous = smooth_points[-1]
            # 이동 평균 방식
            smooth_points.append(previous*factor+point*(1-factor))
        else:# 처음 데이터
            smooth_points.append(point)
    return smooth_points
# average_mae_history[10:] 데이터 모양이 가파른 10개 제외하고 이동평균
smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history)+1), smooth_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation Lost")
plt.show()