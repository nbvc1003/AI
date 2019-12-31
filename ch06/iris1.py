import tensorflow as tf
import pandas as pd
import numpy as np
tf.compat.v1.disable_eager_execution()
keys=['SepalLength','SepalWidth','PetalLength','PetalWidth']
data = pd.read_csv("iris.csv")
# print(data.shape)
# 종류 데이터 중에서 중복을 제거하고 리스트에 담는다
species = list(data['Species'].unique())
# print(species)
# class에 species별로 구별해서 one_hot변경
# lambda 파라미터(x)를 받아서 콜론(:)뒤 결과를 return
data['class'] = data['Species'].map(lambda x:np.eye(len(species))[species.index(x)])
# print(data)
# data중에서 랜덤하게 50개를 추출하여 test
test_set = data.sample(50)
# 테스트 데이터를 제거하고 나머지 훈련데이터
train_data = data.drop(test_set.index)
X = tf.compat.v1.placeholder(tf.float32, [None, 4])
Y = tf.compat.v1.placeholder(tf.float32, [None, 3])
W = tf.Variable(tf.zeros([4, 3]), name = 'weight')
b = tf.Variable(tf.zeros([3], name='bias'))

H = tf.nn.softmax(tf.matmul(X, W) + b)
cost=tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2( logits=H, labels=Y, name='cross_entropy'))
train = tf.compat.v1.train.GradientDescentOptimizer(
    learning_rate=0.5).minimize(cost)
# math.argmax 가장 큰값의 인덱스 번호
accuaracy = tf.reduce_mean(tf.cast(tf.equal(tf.math.argmax(H,1),
                            tf.math.argmax(Y, 1)), tf.float32))
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
# train_data데이터 중에서 class부분만 가져와서 train_set_class
train_set_class = [y for y in train_data['class'].values]
test_set_class = [y for y in test_set['class'].values]
# print(train_set_class)
for i in range(10001):
    t_,a_ = sess.run([train,accuaracy],
            feed_dict={X:train_data[keys].values,Y:train_set_class})
    if i % 100 == 0:
        print(i, a_)
acc = sess.run(accuaracy,feed_dict={X:test_set[keys].values,
                                    Y:test_set_class})
print('test data의 정확도 :',acc)
species = ["setosa","versicolor","virginca"]
sample = data.sample(1)
result = species[np.argmax(sess.run(H, feed_dict={X:sample[keys].values}))]
if result == sample['Species'].values:
    print(("맞췄네"))
else:
    print("틀렸어")