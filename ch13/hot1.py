import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('tem10y.csv', encoding='utf-8')
hot_bool = (df['기온'] > 30) # df['기온'] 값이 30이넘으면 True 인 Series 리턴
# print(type(hot_bool))
print(hot_bool)
hot = df[hot_bool] # 30넘은 데이터만 추출
# print(hot)

# 연도별 30도가 넘는 날짜수
cnt = hot.groupby(['연'])['연'].count() # Series
print(type(cnt),cnt)
cnt.plot()
plt.savefig('temp-over30.png')
plt.show()
