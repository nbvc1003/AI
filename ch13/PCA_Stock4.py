import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime
from sklearn.decomposition import PCA
import numpy as np

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

symbols = [
 "SPASTT01USM661N", # US: 미국
 "SPASTT01JPM661N", # JP: 일본
 "SPASTT01EZM661N", # EZ: 유럽
 "SPASTT01KRM661N", # KR: 한국
]
data = pd.DataFrame()
for sym in symbols:
    # 월별 각국 주가 지표
    temp = web.DataReader(sym, data_source='fred', start=datetime.datetime(1998, 1, 1),
        end=datetime.datetime(2017, 12, 31))
    # print(temp.head())
    data[sym] = temp[sym]
data.columns = ["US", "JP","EZ","KR"] # 컬럼명 변경
# print(data)

# 데이터의 첫번째 행의 값을 100 으로 기준 해서 값을 재설정
# 변동 비율을 확인하기 용이하다.
data = data / data.iloc[0] * 100


# 수익률 데이터
# pct_change() : (현재 데이터 - 이전데이터) / 이전데이터
# resample('A') : D:일 , M: 월, A:년, B: 비지니스데이 -> 그룹핑
# prod() : 그룹별로 전부 곱한다. (앞에서 +1을해줘야 값이 정상적으로 나온다.)
# 앞에서 더해준값을  -1하고  *100 : 퍼샌트 갑이 나온다.
df = ((data.pct_change()+1).resample('A').prod() - 1).T * 100
dftest1 = data.pct_change().resample('A')
print(dftest1.mean())
styles = ['b-','g--','c:','r-'] # 각 그래프선의 스타일 지정

pca2 = PCA(n_components=1) # 주성분을 1개만
w = pca2.fit_transform(df)
df_i = pd.DataFrame(pca2.inverse_transform(w)) # 주성분을 다시 원데이터에 적용해서 (강조시켜서)
df_i.index = df.index
df_i.columns = df.columns
df_i.iloc[:, -10:] # 행은 전부 , 열은 끝 10개
print(df_i)
df_i.T.plot(style = styles)
plt.title("주성분 사용한 20년 수익율 근사치")
plt.xticks(df.columns)
plt.show()

