import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow_core.examples import input_data
tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(777)
# reproducibility
mnist = input_data.read_data_sets ("MNIST_data/", one_hot =True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
drop_prob = 0.3
# dropout ( keep_prob ) rate  0.7 on training,hould be 1 for testing
drop_prob = tf.compat.v1.placeholder(tf.float32)
# input place holders
X = tf.compat.v1.placeholder(tf.float32, [None, 784])
Y = tf.compat.v1.placeholder(tf.float32, [None, 10])
W1 = tf.compat.v1.get_variable("W1", shape=[784, 512],
            initializer=tf.compat.v1.initializers.glorot_normal())
b1 = tf.Variable(tf.random.normal([512]))
_L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(_L1, drop_prob)
W2 = tf.compat.v1.get_variable("W2", shape=[512, 512],
            initializer=tf.compat.v1.initializers.glorot_normal())
b2 = tf.Variable(tf.random.normal([512]))
_L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(_L2, drop_prob )
W3 = tf.compat.v1.get_variable("W3", shape=[512, 512],
            initializer=tf.compat.v1.initializers.glorot_normal())
b3 = tf.Variable(tf.random.normal([512]))
_L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(_L3, drop_prob )
W4 = tf.compat.v1.get_variable("W4", shape=[512, 512],
            initializer=tf.compat.v1.initializers.glorot_normal())
b4 = tf.Variable(tf.random.normal([512]))
_L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(_L4, drop_prob )
W5 = tf.compat.v1.get_variable("W5", shape=[512, 10],
initializer=tf.compat.v1.initializers.glorot_normal())
b5 = tf.Variable(tf.random.normal([10]))
hypothesis = tf.matmul(L4, W5) + b5
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.compat.v1.train.AdamOptimizer(
learning_rate=learning_rate).minimize(cost)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, drop_prob: 0.3}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
        print('Epoch:', '%04d' % (epoch + 1), 'cost =',
              '{:.9f}'.format(avg_cost))
    print('Learning Finished!')
correct_prediction = tf.equal(tf.argmax(hypothesis, 1),
                        tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
    X: mnist.test.images, Y: mnist.test.labels, drop_prob: 0}))