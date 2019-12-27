import tensorflow as tf
import numpy as np

# SIGMOID 함수 : 어떤 값이 들어와도 0 ~ 1 사이의 값이 나온다.
# 데이터값의 차이가 너무 클때 사용한다.

# SIGMOID 함수
def sigmoid(z):
    return 1/(1+np.e**-z)

print(sigmoid(100))
print(sigmoid(0))
print(sigmoid(-50))

print(sigmoid(np.array([100, 0, -10])))