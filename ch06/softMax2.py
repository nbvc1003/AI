import numpy as np

y = [2.0, 1.0, 0.1]

# exp(x) x 이 되는 자연상수 e 의 지수값
print(np.exp(y))
print(np.exp(y)/np.sum(np.exp(y)))# 모든 행열값들의 합이 1인 값으로 변형 시킨다.

