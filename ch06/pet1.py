import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# 열 4개
x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,6,6,6],
          [1,7,7,7]]

# 0 : 고양이, 1: 개, 2:고슴도치
# 열 3개
y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

X = tf.compat.v1.placeholder(tf.float32,[None, 4])
Y = tf.compat.v1.placeholder(tf.float32,[None, 3])
nb_classes = 3
w = tf.Variable(tf.random.normal([4,nb_classes]), name='weight')
b = tf.Variable(tf.random.normal([nb_classes]))

# 2건은 sigmoid, 3건이상은 softmax  
# 3개의 식을 하나로 matmul, softmax : 여러개 데이터 합계가 1인 확율로 계산
hypothesis = tf.nn.softmax(tf.matmul(X,w)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.math.log(hypothesis), axis=1))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(2001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step%200 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

    a1 = sess.run(hypothesis, feed_dict={X:[[1,11,7,9]]})
    # argmax : 값이 가장 큰 데이터 인덱스 번호 a1 데이터 , 1 : 결과값의 차원
    print(a1, sess.run(tf.argmax(a1,1))) # one_hot

    a2 = sess.run(hypothesis, feed_dict={X:[[1,3,4,3]]})
    print(a2, sess.run(tf.argmax(a2,1))) # one_hot

    a3 = sess.run(hypothesis, feed_dict={X:[[1,1,0,1]]})
    print(a3, sess.run(tf.argmax(a3,1))) # one_hot

    a4 = sess.run(hypothesis, feed_dict={X:[[1,1,0,1],
                                            [1,11,7,9],
                                            [1,3,4,3]],
                                         })

    # argmax 가장큰값의 인덱스 
    print(a4, sess.run(tf.argmax(a4,1))) # one_hot


