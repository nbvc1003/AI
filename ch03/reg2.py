import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()

num_poinst = 1000

vector_set = []

for i in range(num_poinst):
    # 평균 0 표준편차 0.55
    x1 = np.random.normal(0.0, 0.55)
    # 기울기가 0.1 절편 0.3
    y1 = x1*0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vector_set.append([x1, y1])

# 결과적으로 조건에 해당하는 랜덤한 절편값을 갖는 함수를 가지고
# 어느정도 크게 벗어 나지 않는 점들의 좌표값을 가져 온다.
# 테스트하기 적당한 무작위 값
x_data = [v[0] for v in vector_set]
y_data = [v[1] for v in vector_set]

# random_uniform x( -1 ~ 1) 모든곳의 확률이 동일한 구조
w = tf.Variable(tf.compat.v1.random_uniform([1], -1.0, 1.0)) # 배열1개 짜리 값 랜덤
b = tf.Variable(tf.zeros([1])) # 배열 1개짜리 값 0
# 변수에 임의의 초기 값을 셋팅해준다.

h = w * x_data + b
loss = tf.reduce_mean(tf.square(h - y_data))

# loss값이 최소값이 되는 값을 찾는 함수
# GradientDescentOptimizer 함수는 주어진 식의 tf.Variable으로 선언된 변수값들을 로직에 맞게 learning_rate 값 단위로 변화 시키면서 찾아 간다.
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
# 결과적으로 찾은 값은 w, b 변수에 저장된다.

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(5):
    sess.run(train) # train 실행
    plt.plot(x_data, y_data, 'ro') # 원데이터
    
    #                      예측된 데이터
    plt.plot(x_data, sess.run(w)*x_data+sess.run(b)) # 잡업된 결과
    plt.show()

sess.close()



