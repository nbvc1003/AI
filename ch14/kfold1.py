from sklearn.model_selection import KFold
from sklearn.datasets import load_boston
import statsmodels.api as sm
import pandas as pd
import numpy as np
boston = load_boston()
dfx = pd.DataFrame(boston.data, columns=boston.feature_names)
dfy = pd.DataFrame(boston.target, columns=['MEDV'])
df = pd.concat([dfx, dfy], axis=1)
scores = np.zeros(5)

# 데이터를 썩어서 5개 랜덤하게 섰었으면 그 다음에는 같은 데이터
cv = KFold(5, shuffle=True, random_state=0)
for i, (idx_train, idx_test) in enumerate(cv.split(df)):
    df_train = df.iloc[idx_train]
    df_test = df.iloc[idx_test]
    model = sm.OLS.from_formula("MEDV ~ " + "+".join(boston.feature_names), data=df_train)
    result = model.fit()
    pred = result.predict(df_test)
    rss = ((df_test.MEDV - pred)**2).sum()
    tss = ((df_test.MEDV - df_test.MEDV.mean())**2).sum()
    rquared = 1 - rss /tss
    scores[i] = rquared
    print("학습 R2 = {:.4f},검증 R2={:,.4f}".format(result.rsquared, rquared))




