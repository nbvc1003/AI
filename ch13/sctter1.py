from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

iris_dataset = load_iris()
X_train, X_test , Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
df = pd.DataFrame(X_train, columns=iris_dataset.feature_names) # iris_dataset 의 키네임을 컬럼명으로 사용
pd.plotting.scatter_matrix(df, c=Y_train, figsize=(8,8),marker='o')
plt.show()
