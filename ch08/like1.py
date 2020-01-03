import tensorflow as tf

x = [[0,1,2],[2,1,0]]

# x와 shape가 같은 1로 채워진 배열을 만든다.
tf.print(tf.ones_like(x))

# x와 shape가 같은 0로 채워진 배열을 만든다.
tf.print(tf.zeros_like(x))

tf.print(zip([1,2,3],[4,5,6]))

# zip 여러개의 배열을 분기문에서 함께 사용가능하도록 해줌..
for x, y in zip([1,2,3],[4,5,6]):
    tf.print(x,y)

for x, y, z in zip([1,2,3],[4,5,6],[7,8,9]):
    tf.print(x,y, z)