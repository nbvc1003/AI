import tensorflow as tf

tf.compat.v1.disable_eager_execution()

x1_data = [73.,93., 89.,96.,73.]
x2_data = [80.,88., 91.,98.,66.]
x3_data = [75.,93., 90.,100.,70.]

y_data = [152., 185.,180., 196., 142.]


# 실 데이터가 들어갈 변수
# placeholder : session.run( feed_dist= {값}) 에서 주어지는 값이 들어가는 장소
x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

# 가중치 w1, w2, w3 
w1 = tf.Variable(tf.random.normal([1]), name='weight1')
w2 = tf.Variable(tf.random.normal([1]), name='weight1')
w3 = tf.Variable(tf.random.normal([1]), name='weight1')
#
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b # 각데이터들을 적용했을때 결과.

cost = tf.reduce_mean(tf.square(hypothesis -y))

# cost값이 가장 작아지는 값을 찾는다.
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(3000):

    #  .run([cost, hypothesis, train]....)
    # 함수는 첫번째 파라메터에 들어가는 변수의 결과 값을 각각 리턴한다.
    # cost_ = cost , hy_ = hypothesis, t_ = train
    cost_, hy_, t_ = sess.run([cost, hypothesis, train], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    if i%20 == 0:
        print(i, 'cost :', cost_, "Predict:\n",hy_)

sess.close()


