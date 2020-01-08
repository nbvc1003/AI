import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
hidden_size = 5 # 임의로 지정 가능 일반적으로 input_dim 크기와 같게
input_dim = 5 #   h,i,e,l,o
batch_size = 1 # hihello 문장 1개
sequence_length = 6 # output_length  ihello
idx2char = ['h','i','e','l','o'] # 인덱스가 오면 해당하는 문자
x_data = [[0,1,0,2,3,3]] # hihell
x_one_hot = [[
            [1,0,0,0,0], # h
            [0,1,0,0,0], # i
            [1,0,0,0,0], # h
            [0,0,1,0,0], # e
            [0,0,0,1,0], # l
            [0,0,0,1,0]  # l
              ]]
y_data = [[1,0,2,3,3,4]] # ihello

X = tf.compat.v1.placeholder(tf.float32, [None, sequence_length, input_dim])
Y = tf.compat.v1.placeholder(tf.int32, [None, sequence_length])

cell = tf.keras.layers.SimpleRNNCell(units=hidden_size)
outputs, _states = tf.compat.v1.nn.dynamic_rnn(cell, X, dtype=tf.float32)
# weights = tf.ones([batch_size, sequence_length])

# 실제 Y값과 비교해서 오차값 # RNN의 cost or loss
sequence_loss = tf.keras.losses.sparse_categorical_crossentropy(Y, outputs, from_logits=True)
loss = tf.reduce_mean(sequence_loss)

train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
prediction = tf.argmax(outputs, axis=2)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(2001):
        l_, _ = sess.run([loss,train], feed_dict={X:x_one_hot, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_one_hot})
        print(i, 'Loss:', l_, ',예측:', result, ' , 실제값:', y_data)

    print(result)

    # idx2char 예측된 인덱스 번호를 문자로 변경
    # squeeze 값이 하나인 차원제거
    result_str = [idx2char[c] for c in np.squeeze(result)]

    print(result_str)
    print('예측 문장 ', ''.join(result_str))


















