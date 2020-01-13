import statsmodels.api as sm
import datetime
import dateutil
import matplotlib.pyplot as plt

# R에 있는 패키지를 불러 온다.
data = sm.datasets.get_rdataset('deaths', "MASS") # MASS 패키지안의 deaths 데이터
df = data.data
print(df.tail())

# 년도.실수 -> 년월일 로변경..
def yearfra2datetime(yearfraction, startyear = 0):
    year = int(yearfraction) + startyear  # 연도
    month = int(round(12*(yearfraction - year))) # 소수점에 * 12 -> 월
    delta = dateutil.relativedelta.relativedelta(months=month)
    date = datetime.datetime(year, 1,1) + delta
    return date

# time 컬럼을 yearfra2datetime 함수에 적용한값으로
df['datetime'] = df.time.map(yearfra2datetime)
print(df.tail())

df.plot(x='datetime',y='value')
plt.title(data.title)
plt.show()


