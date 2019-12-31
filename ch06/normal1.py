import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

# 정규화 함수
def min_max(data):
    min = np.min(data)
    max = np.max(data)
    return (data - min)/ (max - min)


xy = np.array([ [828.659973, 833.450012, 908100, 828.349976, 831.659973],
                [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
                [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
                [816, 820.958984, 1008100, 815.48999, 819.23999],
                [819.359985, 823, 1188100, 818.469971, 818.97998],
                [819, 823, 1198100, 816, 820.450012],
                [811.700012, 815.25, 1098100, 809.780029, 813.669983],
                [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

# 정규화
xy = min_max(xy)

x_data = xy[:,:-1] #  데이터
y_data = xy[:,[-1]] # 예측할 값
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random.normal([4,1]),name= 'weight')
b = tf.Variable(tf.random.normal([1]),name= 'bias')

hypothesis = tf.matmul(X,W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(2001):
    c_, h_, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data,Y:y_data})
    if i % 100 == 0:
        print(i, 'Cost :', c_, '예측 :',h_)

# 데이터의 값 차이가 너무 크면 정규화를 해준다.


