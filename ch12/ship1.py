import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

# 하나의 문자열..
sentence=("if you want to build a ship, don't drum up people together"
 " to collect wood and don't assign them tasks and work, but "
 "rather teach them to long for the endless immensity of the sea.")
char_set = list(set(sentence)) # 중복된 철자를 제거
char_dic = {w:i for i, w in enumerate(char_set)} # 문자 리스트를 딕셔너리로
dataX = []
dataY = []
data_dim = len(char_set) #
hidden_size = len(char_set)
num_classes = len(char_set)
seq_length = 10 # input 단어를 문자 10개씩


for i in range(0, len(sentence) - seq_length):
    # 실제 x_data, y_data 의 갯수
    x_str = sentence[i:i + seq_length]
    y_str = sentence[i+1 : i+seq_length + 1]
    x = [char_dic[c] for c in x_str] # 문자열을 숫자로
    y = [char_dic[c] for c in y_str]
    print(i, x, y)
    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)
X = tf.compat.v1.placeholder(tf.int32, [None, seq_length]) # 입력 10개
Y = tf.compat.v1.placeholder(tf.int32, [None, seq_length]) # 출력도 10개  10개로 10개 찾는다.
X_one_hot = tf.one_hot(X, num_classes) # num_classes :글자 종류수 에 맞게 one_hot 배열로
cell = tf.keras.layers.LSTMCell(units=hidden_size)
# 샐을 여러개를 쌓는다.
cell = tf.keras.layers.StackedRNNCells([cell]*2) # cell은 같은 종류만 가능?
outputs, _states = tf.compat.v1.nn.dynamic_rnn(cell, X_one_hot, dtype=tf.float32)

# RNN의  3차원 데이터를 사용하는데 . softmax를 사용하기 위해 2차원 행열로 변경 한다.
x_for_softmax = tf.reshape(outputs, [-1, hidden_size])
softmax_w = tf.compat.v1.get_variable('sw', [hidden_size,num_classes])
softmax_b = tf.compat.v1.get_variable('sb',[num_classes])

outputs = tf.matmul(x_for_softmax, softmax_w ) +softmax_b
outputs = tf.reshape(outputs,[batch_size, seq_length, num_classes])
sequence_loss = tf.keras.losses.sparse_categorical_crossentropy(Y, outputs, from_logits=True)
loss = tf.reduce_mean(sequence_loss)

train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(1001):
    _, l_, results = sess.run([train, loss, outputs], feed_dict={X:dataX, Y:dataY})
    for j , result in enumerate(results):
        index = np.argmax(result, axis=1)
        # 숫자로 나은 결과를 문자로 변경하여 출력
        # join List 데이터를 문자열로 변경
        print(i, j, ''.join([char_set[t] for t in index]), 1)

results = sess.run(outputs, feed_dict={X:dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:
        print(''.join([char_set[t] for t in index ] ), end='')
    else:
        print(char_set[index[-1]], end='')










