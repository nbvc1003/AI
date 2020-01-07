import numpy as np

a = np.arange(24).reshape(2,3,4)
print(a)
print(a.shape)
print(np.swapaxes(a, 0, 1)) # 0번째와 1번째를 바꾼다. 따라서 shape= (2,3,4) -> (3,2,4) 로바뀐다.
print(np.swapaxes(a, 0, 1).shape)

#  0번째와 2번째를 바꾼다. 따라서 shape= (2,3,4) -> (4,3,2) 로바뀐다.
print(np.swapaxes(a, 0,2))
print(np.swapaxes(a, 0,2).shape)