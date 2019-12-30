import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()

x = np.array([1.0,1.0,2.0])
y = softmax(x)
ration = y
labels = y
plt.pie(ration, labels= labels, shadow=True, startangle=90)
plt.show()

