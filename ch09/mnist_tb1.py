import tensorflow as tf
from tensorflow_core.examples import input_data
tf.compat.v1.disable_eager_execution()
mnsit = input_data.read_data_sets('MNIST_data/', one_hot=True)
nb_classes = 10
X = tf.compat.v1.placeholder(tf.float32, [None, 784])
Y = tf.compat.v1.placeholder(tf.float32, [None, nb_classes])
W = tf.Variable(tf.random.normal([28*28, nb_classes]))
b = tf.Variable(tf.random.normal([nb_classes]))
hypothesis = tf.nn.softmax(tf.matmul(X,W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.math.log(hypothesis), axis=1))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
cost_hist = tf.compat.v1.summary.scalar("cost", cost)
is_correct = tf.equal(tf.math.argmax(hypothesis, 1), tf.math.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
acc_hist = tf.compat.v1.summary.scalar('accuracy', accuracy)
training_epochs = 1 # 데이터 전체를 몇번 실행
batch_size = 10 # 한번에 실행할 단위
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    merged = tf.compat.v1.summary.merge_all()
    writer = tf.compat.v1.summary.FileWriter("E:/nbvc/python/AI/ch09/log/mnist1")
    writer.add_graph(sess.graph)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnsit.train.num_examples/batch_size)
        for i in range(total_batch):# total_batch 만큼 반복
            batch_xs, batch_ys = mnsit.train.next_batch(batch_size)
            c_,_,s_ = sess.run([cost,train,merged], feed_dict={X:batch_xs,Y:batch_ys})

            writer.add_summary(s_, global_step=i)
            avg_cost += c_ / total_batch
    print('accuracy:', sess.run(accuracy, feed_dict={X:mnsit.test.images, Y:mnsit.test.labels}))






