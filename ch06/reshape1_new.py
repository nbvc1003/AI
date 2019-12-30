import tensorflow as tf

t1 = tf.Variable([1,2,3,4,5,6,7,8,9])
print(t1.shape) # [9]
t2 = tf.Variable([
                    [
                        [1,1],
                        [2,2]
                    ],
                    [
                        [3,3],
                        [4,4]
                    ]
                ])
# shape 순서는 바깥쪽에서 안족 순으로 ..
print(t2.shape) # [2,2,2]
t3 = tf.Variable(
                    [
                        [
                            [1,1,1],
                            [2,2,2]
                        ],
                        [
                            [3,3,3],
                            [4,4,4]
                        ],
                        [
                            [5,5,5],
                            [6,6,6]
                        ]
                    ]
                )

print(t3.shape) # [3,2, 3]

t4 = tf.Variable([7])
print(t4.shape)

# -1 은 flatten (행열은 정리 유추해서)
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
tf.print(tf.reshape(t1, [3,3])) # 9열 -> 3 * 3 로 바꾼다.
tf.print(tf.reshape(t2, [2,4])) # 2 * 3열 -> 2 * 4 로 바꾼다.

tf.print(tf.reshape(t3, [-1])) # [-1] : 1차원 으로 알아서 바꾼다.
tf.print(tf.reshape(t3, [2, -1])) # [-1] : 2행 으로 열은 알아서 바꾼다.
tf.print(tf.reshape(t3, [-1, 9]))  # 열은 9 개 행은 알아서
# [[]]은 2개, []속은 3개 , [[]]안에 []는 알아서 만든다.
tf.print(tf.reshape(t3, [2,-1, 3]))  #
tf.print(tf.reshape(t4, []))