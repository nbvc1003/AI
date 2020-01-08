import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

sample = "if you want you"
idx2char = list(set(sample)) # 중복된 문자를 제거하고 List로 변경
# 문자에 키값 부여 딕셔너리로
char2idx = {c: i for i,c in enumerate(idx2char)}

sample_idx = [char2idx[c] for c in sample] # 숫자 list로
x_data = [sample_idx[:-1]] # if you want yo 를 숫자 list로
y_data = [sample_idx[1:]] # f you want you 를 숫자 list로

dic_size = len(char2idx)  # 컬럼갯수 input_dim 중복되지 않는 문자수
hidden_size = len(char2idx) # 제한이 없지만 일반적으로 input_dim 과 같은수로
batch_size = 1
sequence_length = len(sample) - 1 # 문자 전체 길이에서 1글자 제외

X = tf.compat.v1.placeholder(tf.int32, [None, sequence_length])
Y = tf.compat.v1.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X, dic_size)

cell = tf.keras.layers.SimpleRNNCell(units=hidden_size)
outputs, _states = tf.compat.v1.nn.dynamic_rnn(cell, X_one_hot, dtype=tf.float32)
# weight = tf.ones([batch_size, sequence_length])
sequence_loss = tf.keras.losses.sparse_categorical_crossentropy(Y, outputs, from_logits=True)
loss = tf.reduce_mean(sequence_loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
prediction = tf.argmax(outputs, axis=2)
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(3001):
        l_, _ = sess.run([loss, train], feed_dict={X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data}) # 숫자
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, 'loss', l_, ' 예측 :', ''.join(result_str)) # ''.join(result_str) 문자 list 를 한 문자열로














