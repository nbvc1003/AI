import numpy as np

t = np.array([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]])

print(t)
print(t.ndim) # 차춴
print(t.shape) # 형태 : (행 , 열) -> 바깥쪽 부터

t1 = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],
                 [[13,14,15,16],[17,18,19,20],[21,22,23,24]]])

print(t1.shape)