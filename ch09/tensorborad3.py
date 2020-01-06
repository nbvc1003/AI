import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
# tf.compat.v1.set_random_seed(123)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

X = tf.compat.v1.placeholder(tf.float32, [None, 2], name='x-input')
Y = tf.compat.v1.placeholder(tf.float32, [None, 1], name='y-input')

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random.normal([2,2]), name='weight1')
    b1 = tf.Variable(tf.random.normal([2]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1)+b1)

    # layer1 에 대한 기록
    W1_hist = tf.compat.v1.summary.histogram('weight1', W1)
    b1_hist = tf.compat.v1.summary.histogram('bias1', b1)
    layer1_hist = tf.compat.v1.summary.histogram('layer1', layer1)

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random.normal([2,1]), name='weight2')
    b2 = tf.Variable(tf.random.normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2)+b2)

    # layer2 에 대한 기록
    W2_hist = tf.compat.v1.summary.histogram('weight2', W2)
    b2_hist = tf.compat.v1.summary.histogram('bias2', b2)
    hypothesis_hist = tf.compat.v1.summary.histogram('hypothesis', hypothesis)

with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y * tf.math.log(hypothesis) + (1-Y) * tf.math.log(1-hypothesis))
    cost_summ = tf.compat.v1.summary.scalar("cost", cost)

with tf.name_scope("train") as scope:
    train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.02).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_summ = tf.compat.v1.summary.scalar("accuaracy", accuracy)

with tf.compat.v1.Session() as sess:
    merged_summary = tf.compat.v1.summary.merge_all()

    # writer = tf.compat.v1.summary.FileWriter("E:/nbvc/python/AI/ch09/log/sample_3", sess.graph)

    # 위와동일
    writer = tf.compat.v1.summary.FileWriter("E:/nbvc/python/AI/ch09/log/sample_3")
    writer.add_graph(sess.graph)

    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(20001):
        summary_, _ = sess.run([merged_summary, train], feed_dict={X:x_data,Y:y_data})
        writer.add_summary(summary_, global_step=i)
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data,Y:y_data})
    print ('hypothesis :', h, '\n 예측 predicted :', p,'\정확도 accuracy :', a)





