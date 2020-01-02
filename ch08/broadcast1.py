import tensorflow as tf

mat1 = tf.constant([[3.,3.]]) # 1, 2
mat2 = tf.constant([[2.],
                    [2.]]) # 2, 1


tf.print(tf.matmul(mat1, mat2))

# 행열의 연산은 broadcast를 통하여 행열의 크기를 맞춘후 연산
tf.print(mat1 * mat2) # 차원이 빈곳을 기존의 값으로 채워서 각 같은 위치값끼리 연산 
                      
tf.print(mat1 + mat2)
tf.print(mat1 - mat2)
tf.print(mat1 / mat2)

