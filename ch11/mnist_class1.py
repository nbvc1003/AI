import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_core.examples import input_data

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(123)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
learning_rate = 0.001
training_epochs = 15
batch_size = 100

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.compat.v1.variable_scope(self.name):
            # 훈련할때 제외하는 비율 1.0버전에서는 keep_rate
            self.drop_prob = tf.compat.v1.placeholder(tf.float32)
            self.X = tf.compat.v1.placeholder(tf.float32,[None, 784])
            # cnn에서는 4차원이 필요하다 , [이미지수, 가로, 세로, 채널(색)]
            X_img= tf.reshape(self.X, [-1, 28, 28, 1])
            # 결과 데이터 0~9 까지 숫자 따라서 결과는 10가지
            self.Y = tf.compat.v1.placeholder(tf.float32,[None, 10])
            # [가로, 세로, 채널, 출력수]
            W1 = tf.Variable(tf.random.normal([3,3,1,32], stddev=0.01))

            # strides: 크기 4인1 차원리스트.[0], [3] 은 반드시 1. 일반적으로[1], [2] 는 같은 값 사용.
            #                          [batch, 가로, 세로, 깊이]
            L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
            L1 = tf.nn.relu(L1)
            # ksize가 [1,2,2,1]이라는 뜻은 2칸씩 이동하면서 출력 결과를 1개 만들어 낸다는 것이다. 다시 말해 4개의 데이터 중에서 가장 큰 1개를 반환하는 역할을 한다.
            L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            L1 = tf.nn.dropout(L1, self.drop_prob) # 드랍 비율 셋

            #
            W2 = tf.Variable(tf.random.normal([3,3,32,64], stddev=0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
            L2 = tf.nn.relu(L2)
            # ?, 7, 7, 64
            L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            L2 = tf.nn.dropout(L2, self.drop_prob)  # 드랍 비율 셋

            # 3, 3, 입력, 출력
            W3 = tf.Variable(tf.random.normal([3,3,64,128], stddev=0.01))
            L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME')
            L3 = tf.nn.relu(L3)
            # ?, 4, 4, 128  (소수점은 없으므로 4, 4)
            L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            L3 = tf.nn.dropout(L3, self.drop_prob)  # 드랍 비율 셋

            # fully connect로 처리 2차행열 로 변환
            L3_flat = tf.reshape(L3, [-1, 4 * 4 * 128])
            # input 4*4*128 -> output = 625
            W4 = tf.compat.v1.get_variable('w4', shape=[ 4 * 4 * 128, 625], initializer=tf.compat.v1.initializers.glorot_normal())
            b4 = tf.Variable(tf.random.normal([625]))
            L4 = tf.nn.relu(tf.matmul(L3_flat, W4)+b4)
            L4 = tf.nn.dropout(L4,self.drop_prob)

            W5 = tf.compat.v1.get_variable('w5', shape=[625,10], initializer=tf.compat.v1.initializers.glorot_normal())
            b5 = tf.Variable(tf.random.normal([10]))
            self.logits = tf.matmul(L4, W5) + b5
            #  tf.nn.softmax_cross_entropy_with_logits -> 좀더 세련된 방식
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

    def get_accuracy(self, x_test, y_test, drop_prob=0):
        return self.sess.run(self.accuracy, feed_dict={self.X:x_test, self.Y:y_test, self.drop_prob:drop_prob})

    def predict(self, x_test, drop_prob=0):
        return self.sess.run(self.logits, feed_dict={self.X:x_test, self.drop_prob:drop_prob})

    def train(self, x_data, y_data, drop_prob=0.3):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X:x_data, self.Y:y_data, self.drop_prob:drop_prob})


sess = tf.compat.v1.Session()
m1 = Model(sess, 'm1')
sess.run(tf.compat.v1.global_variables_initializer())
print('시작')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c_, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c_/total_batch
    print('epoch :', epoch,' cost :{:.5f}'.format(avg_cost))
    
print('작업완료')
print('정확도 :', m1.get_accuracy(mnist.test.images, mnist.test.labels))















