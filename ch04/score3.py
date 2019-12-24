import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np

# ' # 라인은 무시됨
data = np.loadtxt('data-01-test-score.csv', delimiter=',',dtype=np.float)

print(data.shape)
# 훈련용 실데이터
x_train = data[:,:-1] # 입력 데이터
y_train = data[:,[-1]] # 결과 데이터

x_test = [[20, 40, 50],
          [90,88, 80]]  # 회귀직선을 이용하여 예측용 더미 결과는 모른다.

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# W는 X의 열을 행으로 Y의 열을 갯수로 하는 행열
W = tf.Variable(tf.random.normal([3,1]))
b = tf.Variable(tf.random.normal([1])) # 초기 값 무작위 값 셋팅 0.0 셋팅해도 무관
hypothesis = tf.matmul(X, W) + b
loss = tf.reduce_mean(tf.square(hypothesis-Y)) # loss == cost
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# 지도학습법에 따른 회귀직선을 구하는 훈련
for i in range(1001):
    t_, w_, l_, h_ = sess.run([train,W,loss, hypothesis], feed_dict={X:x_train, Y:y_train})
    if i%20 == 0:
        print(i, l_, h_)

# 예측 for 문종료후 훈련에서 얻능 W와 b값을 활용
h = sess.run(hypothesis, feed_dict={X:x_test}) # 값을 주지 않은 데이터는 이전 데이터를 사용한다.
print('예측 마지막 시험 점수 :', h)
print('예상 점수 : ', sess.run(hypothesis, feed_dict={X:[[100, 70, 90]]}))
print('예상 점수 : ', sess.run(hypothesis, feed_dict={X:[[60, 70, 80],[90,100,87]]}))
sess.close()









