from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# 암환자 데이터
cancer = load_breast_cancer()
# 훈련데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify= cancer.target, random_state=1)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

# max_depth=4 4단계 까지만 의사결정 질문
tree = DecisionTreeClassifier(max_depth=4,  random_state=0)
tree.fit(X_train, y_train)
print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))


# import matplotlib.pyplot as plt
# import mglearn
#
# mglearn.plots.plot_animal_tree()




