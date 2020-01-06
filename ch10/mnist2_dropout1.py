import tensorflow as tf
from tensorflow_core.examples import input_data
tf.compat.v1.disable_eager_execution()
mnsit = input_data.read_data_sets('MNIST_data/', one_hot=True)
nb_classes = 10

drop_prob = tf.compat.v1.placeholder(tf.float32) # 버리는 비율

X = tf.compat.v1.placeholder(tf.float32, [None, 784])
Y = tf.compat.v1.placeholder(tf.float32, [None, nb_classes])

W1 = tf.compat.v1.get_variable('w1', shape=[784,512],
                               initializer=tf.compat.v1.initializers.glorot_normal())
b1 = tf.compat.v1.get_variable('b1', shape=[512])
_layer1 = tf.nn.relu(tf.matmul(X,W1) + b1)
layer1 = tf.nn.dropout(_layer1, drop_prob) # 버려질 비율을 설정

W2 = tf.compat.v1.get_variable('w2', shape=[512,512],
                               initializer=tf.compat.v1.initializers.glorot_normal())
b2 = tf.compat.v1.get_variable('b2', shape=[512])
_layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
layer2 = tf.nn.dropout(_layer2, drop_prob)

W3 = tf.compat.v1.get_variable('w3', shape=[512,512],
                               initializer=tf.compat.v1.initializers.glorot_normal())
b3 = tf.compat.v1.get_variable('b3', shape=[512])
_layer3 = tf.nn.relu(tf.matmul(layer2,W3) + b3)
layer3 = tf.nn.dropout(_layer3, drop_prob)

W4 = tf.compat.v1.get_variable('w4', shape=[512,512],
                               initializer=tf.compat.v1.initializers.glorot_normal())
b4 = tf.compat.v1.get_variable('b4', shape=[512])
_layer4 = tf.nn.relu(tf.matmul(layer3,W4) + b4)
layer4 = tf.nn.dropout(_layer4, drop_prob)

W5 = tf.compat.v1.get_variable('w5', shape=[512,512],
                               initializer=tf.compat.v1.initializers.glorot_normal())
b5 = tf.compat.v1.get_variable('b5', shape=[512])
_layer5 = tf.nn.relu(tf.matmul(layer4,W5) + b5)
layer5 = tf.nn.dropout(_layer5, drop_prob)


W6 = tf.compat.v1.get_variable('w6', shape=[512,512],
                               initializer=tf.compat.v1.initializers.glorot_normal())
b6 = tf.compat.v1.get_variable('b6', shape=[512])
_layer6 = tf.nn.relu(tf.matmul(layer5,W6) + b6)
layer6 = tf.nn.dropout(_layer6, drop_prob)


W7 = tf.compat.v1.get_variable('w7', shape=[512,512],
                               initializer=tf.compat.v1.initializers.glorot_normal())
b7 = tf.compat.v1.get_variable('b7', shape=[512])
_layer7 = tf.nn.relu(tf.matmul(layer6,W7) + b7)
layer7 = tf.nn.dropout(_layer7, drop_prob)

W8 = tf.compat.v1.get_variable('w8', shape=[512,nb_classes],
                               initializer=tf.compat.v1.initializers.glorot_normal())
b8 = tf.compat.v1.get_variable('b8', shape=[nb_classes])
hypothesis = tf.nn.softmax(tf.matmul(layer7,W8) + b8)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.math.log(hypothesis), axis=1))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# cost_hist = tf.compat.v1.summary.scalar("cost", cost)
is_correct = tf.equal(tf.math.argmax(hypothesis, 1), tf.math.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
# acc_hist = tf.compat.v1.summary.scalar('accuracy', accuracy)
training_epochs = 10 # 데이터 전체를 몇번 실행
batch_size = 10 # 한번에 실행할 단위
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnsit.train.num_examples/batch_size)
        for i in range(total_batch):# total_batch 만큼 반복
            batch_xs, batch_ys = mnsit.train.next_batch(batch_size)
            c_,_ = sess.run([cost,train], feed_dict={X:batch_xs,Y:batch_ys, drop_prob:0.3})

            avg_cost += c_ / total_batch
        print('epoch :' ,epoch, ' avg_cost :', avg_cost)

    # test시에는 drop_prob 값을 0으로 해서 버리지 않고 전부 테스트
    print('accuracy:', sess.run(accuracy, feed_dict={X:mnsit.test.images, Y:mnsit.test.labels, drop_prob:0}))






