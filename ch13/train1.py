from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
# 자동으로 train 75% , test 25% 할당
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print('X_train크기 :{}'.format(X_train.shape))
print('X_test크기 :{}'.format(X_test.shape))
print('Y_train크기 :{}'.format(Y_train.shape))
print('Y_test크기 :{}'.format(Y_test.shape))





