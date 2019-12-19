import tensorflow as tf

@tf.function
def multi(x,y):
    return x*y

tf.print(multi(3.0, 5.0))