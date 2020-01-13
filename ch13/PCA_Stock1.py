import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime

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

# print(data)

styles = ['b-','g--','c:','r-'] # 각 그래프선의 스타일 지정
data.plot(style=styles)
plt.title("세계 주요국의 20년간 주가")
# plt.show()

# pct_change() : (현재 데이터 - 이전데이터) / 이전데이터
# resample('A') : D:일 , M: 월, A:년, B: 비지니스데이 -> 기준 정렬
df = ((data.pct_change()+1).resample('A').prod() - 1).T * 100

# dftest1 = data.pct_change().resample('A').prod()
# print(dftest1.iloc[:5])











