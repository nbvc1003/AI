from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import statsmodels.api as sm
boston = load_boston()
dfx = pd.DataFrame(boston.data, columns=boston.feature_names) # feature_names : 컬럼명
dfy = pd.DataFrame(boston.target, columns=['MEDV'])
df = pd.concat([dfx, dfy], axis=1)# axis=1 옆으로 붙인다.

N = len(df)
ration = 0.7
np.random.seed(123)

# 훈련데이터 생성
idx_train = np.random.choice(np.arange(N), np.int(ration*N)) # 랜덤으로 np.int(ration*N)개수만큼 선택

# 훈련데이터가 제외된 데이터를 테스트 데이터로 사용..
idx_test = list(set(np.arange(N)).difference(idx_train)) #

df_train = df.iloc[idx_train] # 훈련데이터
df_test = df.iloc[idx_test] # 테스트데이터
#                               종속변수       종속변수(MEDV를 제외한 모든 컬럼
model = sm.OLS.from_formula('MEDV ~ ' + "+".join(boston.feature_names), data=df_train)

result = model.fit()
print(result.summary())

pred = result.predict(df.test)
rss = ((df_test.MEDV - pred) **2).sum()
tss = ((df_test.MEDV - df_test.MEDV.mean())**2).sum()
rsquares = 1 - rss / tss
print(rsquares) # 결정계수
