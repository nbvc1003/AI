import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

wine = pd.read_csv('winequality-white.csv', sep=';')
y = wine['quality'] # 품질만
X = wine.drop('quality', axis=1) # 품질 제거된 나머지

x_train,x_test, y_train,y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(x_train, y_train)

# 평가
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답율 :",  accuracy_score( y_test,y_pred))
