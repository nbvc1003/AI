from numpy import loadtxt
from xgboost import XGBClassifier, plot_importance
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics
cancer = load_breast_cancer()
y = cancer['target']
X = cancer['data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape, X_test.shape)
# 결정트리 500개
xgb = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=4)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print('정답율 :', metrics.accuracy_score(y_test, xgb_pred))

