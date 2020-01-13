import statsmodels.api as sm


data = sm.datasets.get_rdataset('Titanic', package='datasets')

df = data.data
print(df.tail())

print(data.__doc__[:1005])
