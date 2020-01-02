import numpy as np

t = np.array([0.,1.,2.,3.,4.,5.,6.])
print(t)
print(t.ndim)
print(t.shape)
print(t[0],t[1],t[2])

print(t[2:5], t[-1])
# 2 : 2열 앞까지 , 3: 3열 부터 끝까지
print(t[:2], t[3:])



