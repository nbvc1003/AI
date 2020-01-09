from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris_dataset = load_iris()

#  random_state=0 시드값 0이면 시드값 없음
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# 최근접 알고리즘..
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)

X_new = np.array([[5., 2.9, 1., 0.2]])
prediction = knn.predict(X_new)
print(prediction)
print('예측한 데이터 : {}'.format(iris_dataset['target_names'][prediction]))
