import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.InteractiveSession()

t = np.array([[[0,1,2]
               [3,4,5]],  # 0, 1,2

              [[6,7,8],
               [9,10,11]]]) # 2, 2, 3

print(t.shape)
t1 = tf.transpose(t,[1,0,2])
print(t1.eval())
t2 = tf.transpose(t1,[1,0,2])
print(t2.eval())
t3 = tf.transpose(t2,[1,2,0]) # 2, 3,2
print(t3.eval())
t4 = tf.transpose(t3,[2,0,1]) # 2,2,3
print(t4.eval())

sess.close()
#  각차원의 인덱스를 0, 1, 2 ...으로 지정하고
# transpose 함수 내에서 [1,2,0] 와같이 각 차원의 순서를 [0,1,2] -> [1,0,2] or [2,0,1] 과같이 바꿔 주면
# 각 인덱스의 차원의 배열크기 가 다른 차원의 크기로 변경된다. 따라서 (2,2,3) -> (3,2,2) 와같이 차원의 배열값이 바뀐디ㅏ.


