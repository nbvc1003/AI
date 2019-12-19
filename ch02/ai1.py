import tensorflow as tf
tf.compat.v1.disable_eager_execution()

x_train = [1,2,3] # 훈련데이타
y_train = [3,4,7] # 답 label(라벨)

# 초기값을 평균0 표준편차 1 인 값 랜덤으로 주고 변수명을 명명함..
w = tf.Variable(tf.compat.v1.random_normal([1]), name='weight')
b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')

#회귀 직선이라고 가정 (곡선도 가능)
hypothesis = x_train * w + b

# reduce_mean  : 2개씩을 선택해서 평균을 내면서 개수를 줄여가는 방법
# square : 제곱 ,
cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # (예측-실제값)**2 한값을 평균

# GradientDescentOptimizer : 경사를 따라서 w, b값을 0.01씩변경하며 최소값을 찾는다.
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(2001):
    sess.run(train)
    if i% 20 == 0:
        print(i, sess.run(cost), sess.run(w), sess.run(b))

## 위 공식의 설명
# x_train, y_train(답)
# x_train와 y_train 값의 관계가 직선방정식을 갖
# x_train값을 조금씩 바꿔 주면서  y_train와 가장 가까워지는 w, b 값을 찾는다.


sess.close()

