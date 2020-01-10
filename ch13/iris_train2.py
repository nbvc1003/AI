import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

csv = pd.read_csv('iris.csv')
# 종목을 제외한 필요한 데이터 컬럼만 가져 온다.
csv_data = csv[['SepalLength','SepalWidth','PetalLength','PetalWidth']]
csv_label = csv['Species']
train_data, test_data, train_label, test_label = train_test_split(csv_data, csv_label, train_size= 0.5, test_size=0.5)
print(len(train_data), len(test_data))

clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)
ac_score = metrics.accuracy_score(test_label, pre)
print('정답율 :', ac_score)








