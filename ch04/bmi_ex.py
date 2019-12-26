import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()

# 몸무게 = bmi * 키 **2
x_data = [1.60, 1.70, 1.80 ] # 키
y_data = [55, 60, 65] # 몸무게
x_test = [1.50, 1.60, 1.90] # 키일때 몸무게 예측

X = tf.compat.v1.placeholder(tf.float32)
Y = tf.compat.v1.placeholder(tf.float32)

W = tf.Variable(1.0, tf.float32) # 탠서플로에서 바꿔주는 변수들
b = tf.Variable(1.0, tf.float32)

hypothesis = W * X**2 # 몸무게를 구하는 예측식
loss = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
w_val = []
loss_val = []
eporch = 1001 # 훈련횟수
for i in range(eporch):
    t_, w_, l_ = sess.run([train, W,loss], feed_dict={X:x_data, Y:y_data})
    w_val.append(w_)
    loss_val.append(l_)

h = sess.run(hypothesis, feed_dict={X:x_test})
print('예측 몸무게:' , h)

plt.plot(w_val, loss_val, 'o')
plt.show()
sess.close()


