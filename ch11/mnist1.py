from tensorflow_core.examples import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
img = mnist.train.images[0].reshape(28, 28)
# cmap ( color map)
plt.imshow(img, cmap='gray') # cmap='gray' 숫자가 크면 밝고 작으면 어둡게
plt.show()