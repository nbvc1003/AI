import numpy as np

num_poinst = 1000

vector_set = []

for i in range(num_poinst):
    # 평균 0 표준편차 0.55
    x1 = np.random.normal(0.0, 0.55)
    # 기울기가 0.1 절편 0.3
    y1 = x1*0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vector_set.append([x1, y1])

# 결과적으로 조건에 해당하는 랜덤한 절편값을 갖는 함수를 가지고
# 어느정도 크게 벗어 나지 않는 점들의 좌표값을 가져 온다.
# 테스트하기 적당한 무작위 값
x_data = [v[0] for v in vector_set]
y_data = [v[1] for v in vector_set]

import matplotlib.pyplot as plt

plt.plot(x_data, y_data, 'ro', label= 'Origina Data')
plt.legend()
plt.show()