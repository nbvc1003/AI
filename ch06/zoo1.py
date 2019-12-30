import tensorflow as tf
import  numpy as np
tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(123)

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, :-1] # 마지막 열 제외
y_data = xy[:,[-1]] # 마지막 열만

nb_classes = 7 # 결과 종류가 7가지 0~6
X = tf.compat.v1.placeholder(tf.float32,[None, 16])
Y = tf.compat.v1.placeholder(tf.int32,[None, 1])
Y_one_hot = tf.one_hot(Y, nb_classes) # 7열중 하나만 1 나머지는 0
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # 열은 7개로 변경

# reshape 으로 바뀌기 때문에 아래 와 같이 nb_classes 를 쓴다.
w = tf.Variable(tf.random.normal([16,nb_classes]), name='weight')
b = tf.Variable(tf.random.normal([nb_classes]), name='bias')

logits = tf.matmul(X,w) + b
hypothesis = tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels = Y_one_hot))

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.math.argmax(hypothesis, 1) # 가장큰값의 인덱스

# 예측한 값 prediction 이 실데이터 tf.math.argmax(Y_one_hot,1) 와 비교해서 맞으면 1 다르면 0
correct_prediction = tf.equal(prediction, tf.math.argmax(Y_one_hot,1)) #
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(2001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if i % 100 ==0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X:x_data, Y:y_data})
            print('loss :',loss,' acc',acc)

    pred = sess.run(prediction, feed_dict={X:x_data})
    for p, y in zip(pred, y_data.flatten()):
        print( p==int(y), p, int(y))






