import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

# 데이터의 값들의 차이가 너무 클 경우
# 데이터 값의 크기를 0~1 사이의 값으로 정규화 해준다.
def normalize(input):
    max = np.max(input, axis=0)
    min = np.min(input, axis=0)
    out = (input - min) / (max - min)
    return out

x = np.array([[50,15],[40,20],[10,5],[20,10],[45,22],[15,13]])  # 길이 , 무게
y = np.array([[0],[0],[1],[1],[0],[1]])  # 0: 삼치, 1: 꽁치

x = normalize(x)
y = normalize(y)

testM = 2
m = len(x) - testM

# 훈련데이터
x_train = x[ : m, : ]
y_train = y[ : m, : ]

x_test = x[m:, :]
y_test = y[m:, :]

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.ones([2,1]))
b = tf.Variable(0.0) #== tf.random.normal([1]))


# 계산한 결과 값을 0~1 사이의 값으로 바꿔준다. -> 극단값의 영향력 축소
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)
loss = -tf.reduce_mean(Y * tf.math.log(hypothesis) +
                       (1-Y)* tf.math.log(1-hypothesis))


train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(10001):
    t_, w_, l_, h_, b_ = sess.run([train, W, loss, hypothesis, b], feed_dict={X:x_train, Y:y_train})
    if i%100 == 0:
        print(i, 'loss :',l_,' h :',h_, 'w :', w_, 'b :',b_)

# 훈련데이터를 기준(w, b) 테스트를 진행
predict = sess.run(predicted, feed_dict={X:x_test, Y:y_test})
print('결과 :' , predict)
sess.close()








