import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# 단위를 맞추기 위해서 정규화 : 최대 최소 이용한 정규화
# input 배열의 값이 단위가 너무 차이가 날경우 값이 비슷해지도록 정규화
def normalize(input):
    max = np.max(input, axis=0) # axis=0 열단위로 max 값 찾는다.
    min = np.min(input, axis=0) #
    out = (input - min)/(max - min)
    print('max =', max, ', min =', min)
    return out

data = np.loadtxt('data-02-stock_daily.csv',dtype=np.float32, delimiter=',')
print(type(data),data.shape)

x_data = data[:, :-1] # 마지막열빼고 나머지
y_data = data[:,[-1]] # 마지막열

x_data = normalize(x_data) # 데이터의 정규와 -> 열별로 데이터 값이 비슷해지게
y_data = normalize(y_data)

print(data.shape, x_data.shape, y_data.shape)

testM = 10 # 테스트용으로 10건
m = len(x_data) - testM
# 훈련데이터와 테스트 데이터는 분리(다른 데이터)
x_train = x_data[:m, :] # 행은 m 까지만, 열은 전부
y_train = y_data[:m, :] # 행은 m 까지만, 열은 전부

x_test = x_data[m:,:] # 행은 m부터 끝까지 열은 전부
y_test = y_data[m:,:] # 행은 m부터 끝까지 열은 전부

X = tf.compat.v1.placeholder(tf.float32, shape=[None,4])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])
# X의 열을 행의 숫자로 Y의 열의 개수를 열의 개수로
W = tf.Variable(tf.ones([4,1])) # 탠서플로에서 바꿔주는 변수들
# W의 열의 갯수
b = tf.Variable(tf.zeros([1])) # 탠서플로에서 바꿔주는 변수들
# X1 W1 + X2 W2 + X3 W3 + b
hypothesis = tf.matmul(X, W) + b
loss = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
eporch = 10001 # 훈련횟수
for i in range(eporch):
    t_, w_, l_, h_ = sess.run([train, W,loss,hypothesis], feed_dict={X:x_train, Y:y_train})
    if i%20 ==0:
        print(i, 'loss =', l_)
h = sess.run(hypothesis, feed_dict={X:x_test})
print('예측 종가:' , h)
print('실제 종가 :', y_test)

sess.close()

# volume 값이 주가 값과 차이가 너무 많이 나면 계산이 안됨 
# 따라서 표준화가 필요하다. 
# 






