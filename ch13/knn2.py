from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
iris_dataset = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
print("예측값 :{}".format(y_pred))
print("정확도 : {:,.2f}".format(np.mean(y_pred==Y_test)))
print("정확도 : {:,.2f}".format(knn.score(X_test, Y_test))) # knn.score 정확도 계산 함수 ..
