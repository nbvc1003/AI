import numpy as np
a = np.arange(15).reshape(3,5) # 3행5열 로 변경
print(a)

# 행과 열을 바꾼다.
print(a.T)
print(np.transpose(a))
print(np.swapaxes(a,0,1)) # 특정 차원의 값을 서로 바꿔준다.

