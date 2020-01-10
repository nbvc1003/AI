from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

tbl = pd.read_csv('bmi.csv')
label = tbl['label']
# 단위를 유사 하게
w = tbl['weight'] /100 # 몸무게 최대 100
h = tbl['height'] / 200 # 키 최대 200
wh = pd.concat([w, h], axis=1)
# 훈련데이터와 테스트 데이터를 75/ 25
data_train, data_test, lable_train, label_test = train_test_split(wh, label)
clf = svm.SVC()
clf.fit(data_train, lable_train)
predict = clf.predict(data_test)
ac_score = metrics.accuracy_score(label_test, predict)
print('정답율:', ac_score)
cl_report = metrics.classification_report(label_test, predict)
print('보고서 \n', cl_report)

