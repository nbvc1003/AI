import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

mr = pd.read_csv("mushroom.csv", header=None)
label = []
datas = []
attr_list = []

for i, row in mr.iterrows():
    label.append(row.iloc[0])
    row_data = []
    for v in row.iloc[1:]:
        row_data.append(ord(v)) # 문자를 acsii 코드로 변경
    datas.append(row_data)
data_train, data_test, label_train, label_test = train_test_split(datas, label)
clf = RandomForestClassifier()
clf.fit(data_train, label_train)
predict = clf.predict(data_test)
ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print("정답율 :", ac_score)
print("보고서  :", cl_report)





