import random

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_core.examples import input_data

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(123)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10 # 0~ 9 : 10열

X = tf.compat.v1.placeholder(tf.float32, [None, 784]) # 784 = 28 * 28
Y = tf.compat.v1.placeholder(tf.float32, [None, nb_classes])
W = tf.Variable(tf.random.normal([784, nb_classes]))
b = tf.Variable(tf.random.normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.math.log(hypothesis),axis=1))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 예측배열의 가장큰 값과 실제 정답 배열의 가장큰값이 같은지 비교
# 예측확율이 가장 높은 열과 실제 값에서 열값이 1인 열번호
is_correct = tf.equal(tf.math.argmax(hypothesis, 1), tf.math.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


training_epochs = 15 # 15회 반복 : 훈련의 반복 횟수 를 지정
batch_size = 100 # 한번에 처리하는 묶음 단위 묶음단위로 처리하고 한번에 답을 확인하여 속도를 증가 시킨다.

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        # mnist.train.num_examples : 훈련 데이터 갯수
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c_ , _ = sess.run([cost, train], feed_dict={X:batch_xs, Y:batch_ys})

        avg_cost += c_ / total_batch # 훈련결과 평균?
        print(epoch, 'Cost :', avg_cost)  # 훈련
    print('정확도 :', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
    r = random.randint(0, mnist.test.num_examples -1)
    print('Label :', sess.run(tf.math.argmax(mnist.test.labels[r:r+1],1)))
    print('예측 :',sess.run(tf.math.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys')
    plt.show()










