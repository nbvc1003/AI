import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
wine = pd.read_csv("winequality-white.csv", sep=';')
y = wine['quality']
x = wine.drop('quality', axis=1)

#
# y 레이블 변경 1~9 등급  => 1~3등급
newlist = []
for i in list(y):
    if i<= 4:
        newlist += [0]
    elif i <= 7:
        newlist += [1]
    else:
        newlist += [2]


y = newlist
print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print('정답율 :', accuracy_score(y_test,y_pred))




