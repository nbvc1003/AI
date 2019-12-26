import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(123)
# x와 y값이 랜덤한 data로 선형 데이터 생성
def make_random_data():
    # - 2 ~ 2 사이 데이터 랜덤 200 개
    x = np.random.uniform(low=-2, high=2, size=200)
    y = []
    for t in x:
        #평균이 0 scale = 표준편차 인 정규분포 데이터
        r = np.random.normal(loc= 0.0, scale=(0.5+t*t/3),size=None) # scale=(0.5+t*t/3) -> 표준편차를 계속 바꿔주기 위함
        y.append(r)
    # 전체적인 경향은 기울기 1.726과 절편 -0.84 인직선그래프를 따르지만 랜덤한 값을 만들기위한 목적
    return x, 1.726*x - 0.84 + np.array(y)

# x, y -> 전체적인 경향은 기울기 1.726과 절편 -0.84 인직선그래프를 따르지만 랜덤한 값
x,y = make_random_data()
x_train, y_train = x[:150], y[:150] # 훈련데이터로 150건
x_test, y_test = x[150:], y[150:] # 테스트 데이터는 50건
model = tf.keras.Sequential()

# 받는 데이터를 선언
# input_dim : x데이터의구성 , units : 나가는 데이터
model.add(tf.keras.layers.Dense(units=1, input_dim=1)) # 들어오는 데이터가 1 차원 , 나가는 데이터가 1
model.summary()

# sgd : Stochkcastic Gradient Descent :경사 하강법
# mse : Mean Square Error
model.compile(optimizer='sgd',loss='mse')

# epochs=300 훈련횟수 
# validation_split=0.3 : 으로 주어진 데이터중 다시 30%를 검증용으로 분리하여 사용후 다시 계산을 수정
# history 작업된 결과가 저장
history = model.fit(x_train, y_train, epochs=300, validation_split=0.3)

epoch = np.arange(1, 301) # 훈련횟수

plt.plot(epoch, history.history['loss'], label='Traning loss') # 훈련데이터의 loss
plt.plot(epoch, history.history['val_loss'], label='Validation loss') # 점검용 데이터의 loss
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# 1.0 과의 차이
# hypothesis 자체를 예측해서 만들어준다.
# 유저는 입력될 훈련데이터와 출력 데이터 형식을 선언해주고
# 훈련, 결과 데이터를 넣어주면 알아서   hypothesis 을 예측해준다.





