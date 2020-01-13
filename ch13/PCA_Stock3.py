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
# resample('A') : D:일 , M: 월, A:년, B: 비지니스데이 -> 기준 정렬
df = ((data.pct_change()+1).resample('A').prod() - 1).T * 100
dftest1 = data.pct_change().resample('A')
print(dftest1)
styles = ['b-','g--','c:','r-'] # 각 그래프선의 스타일 지정

pca2 = PCA(n_components=1) # 주성분을 1개만
w = pca2.fit_transform((df))
m = pca2.mean_ # 모든 공통 요인의  평균값
p1 = pca2.components_[0] # 주성분

# 그래프의 x축값 샛팅, 원래 있던 날짜 인덱스가 사라졌기 때문에 그래프에 사용하기 위해서 다시 생성
# 1998 ~ 2017   20 등분
xrange = np.linspace(1998, 2017, 20, dtype=int)
for i in np.linspace(0, 100, 5): # 0~ 100, 5등분
    print(i)
    plt.plot(xrange, pca2.mean_ + p1*i) # 주성분 반영을 단계적으로 상승
print(type(pca2.mean_))
print(pca2.mean_)
plt.plot(xrange, pca2.mean_+p1*100, label="주성분 100배 수익율")
plt.plot(xrange, pca2.mean_, "ko-", lw=5, label="평균 수익율") # lw (line width)
plt.title("주 성분 크기에 따른 수익율 변화")
plt.legend()
plt.show()

df_w = pd.DataFrame(w)
df_w.index = df.index
df_w.ccolumns=["주성분 비중"]
print(df_w)
