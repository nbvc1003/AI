import tensorflow as tf

tf.compat.v1.disable_eager_execution()

x_data = [1.,2.,3.]
y_data = [1.,2.,3.]

w = tf.Variable(tf.random.uniform([1], name='whight')) #
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
hypothsis = w * x

cost = tf.reduce_mean(tf.square(hypothsis - y))

# w의 변화하는 값 a : 0.1 ,
descent = w - tf.multiply(0.1, tf.reduce_mean(
            tf.multiply(
                # 0.1 * ((wx - y) * x  의 평균)
                tf.multiply(w,x)-y, x)))

update = w.assign(descent) # 변경된 값을 w 에 반영..

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(20):
    sess.run(update, feed_dict={x:x_data, y:y_data})
    print(i, sess.run(cost, feed_dict={x:x_data, y:y_data}), sess.run(w))

sess.close()


# 미분값이 작아지는 방향으로 변한다는 내용을 보여줌..
