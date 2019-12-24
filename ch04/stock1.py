import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

data = np.loadtxt('data-02-stock_daily.csv',dtype=np.float32, delimiter=',')
print(data.shape)

x_data = data[:,:-1]
y_data = data[:,[-1]]
print(data.shape, x_data.shape, y_data.shape)

