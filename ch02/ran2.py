import tensorflow as tf

with tf.compat.v1.Session() as sess:
    # 평균0, 표준편차 0.35 , 인 784 행, 200열 랜덤값 테이블
    x = tf.Variable(tf.compat.v1.random.normal([784,200], stddev=0.35))
    y = tf.Variable(x + 3)
    sess.run(tf.compat.v1.global_variables_initializer())
    print(sess.run(y))
    
