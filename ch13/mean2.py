import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('tem10y.csv', encoding='utf-8')
g = df.groupby(['월'])['기온'] # 같은 '월'별로 묶어서 기온값만 가져온다.
print(df.groupby(['월'])) # -> DataFrameGroupBy
# g == SeriesGroupBy
gg = g.sum() /g.count()
print(gg)
gg.plot()
plt.savefig('tem_mong_avg.png')
plt.show()


