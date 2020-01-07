from tensorflow_core.examples import input_data
import matplotlib.pyplot as plt

# tf.data.Dataset 을 사용하여 데이터를 읽어온다.
#https://hiseon.me/data-analytics/tensorflow/tensorflow-dataset/

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# train-images-idx3-ubyte.gz  : 숫자이미지 를 1, 784 배열로 만든 데이터
# train-labels-idx1-ubyte.gz :
batch_xs, batch_ys = mnist.train.next_batch(1) # 1개를 가져 온다.

# print(type(batch_xs))
print(batch_xs.shape)
print(batch_ys.shape)
print(batch_ys)

# batch_xs 행 28, 열 28 로 재구성하여 이미지 모양으로 보여줘라.
plt.imshow(batch_xs.reshape(28, 28), cmap='Greys')
plt.show()