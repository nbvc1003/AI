from sklearn.datasets import load_iris
iris_dataset = load_iris()


# 컬럼이름을 확인
print("iris_dataset의 키: {}".format(iris_dataset.keys()))

# 키 이름으로 정보 추출
print(iris_dataset['DESCR'][:193] +"\n...")

print("타깃의 이름: {}".format(iris_dataset['target_names']))
print("특성의 이름: {}".format(iris_dataset['feature_names']))
print("data의 타입: {}".format(type(iris_dataset['data'])))
print("data의 크기: {}".format(iris_dataset['data'].shape))
print("data의 처음 다섯행:\n{}".format(iris_dataset['data'][:5]))
print("target의타입: {}".format(type(iris_dataset['target'])))
print("target의 크기: {}".format(iris_dataset['target'].shape))
print("타깃:\n{}".format(iris_dataset['target']))