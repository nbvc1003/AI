from sklearn.linear_model import LinearRegression

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('tem10y.csv', encoding='utf-8')
train_year = (df['연'] <= 2015 )
test_year = (df['연'] >= 2016)
interval = 6


# data 배열을 interval 간격으로 x 와 y 배열을 쌍으로 구성..
def make_data(data):
    x = []
    y = []
    temps = list(data['기온'])
    for i in range(len(temps)):
        if i < interval:
            continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return  (x,y)



train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])
# print(train_x)
# print("=================================================")
# print(train_y)

# 선형회귀
lr = LinearRegression(normalize=True)

lr.fit(train_x, train_y) # 학습
pre_y = lr.predict(test_x) # 예측

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(test_y, c='r')
plt.plot(pre_y, c='b')
plt.show()
diff_y = abs(pre_y - test_y)
print("평균 :", sum(diff_y)/len(diff_y))
print("최대 :", max(diff_y))