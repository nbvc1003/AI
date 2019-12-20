import tensorflow as tf

c1 = tf.constant([1,2,3]) # [1,2,3] 의값 설정
c2 = tf.constant(-7, shape=[2,3]) # -7로 2,3 배열을 채운다.
r1 = tf.range(2, 12, 3) # 2부터 12까지 3씩 증가해서 값을 채움. 디폴트 1씩증가
tf.print(c1)
tf.print(c2)

tf.print(r1)