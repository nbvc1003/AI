from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import mglearn
import matplotlib.pyplot as plt

# random_state 랜덤 시드 여부
X, y = make_blobs(random_state=1)
kmeans = KMeans(n_clusters=3) # 3개의 그룹으로 분리된 데이터
kmeans.fit(X)  # 훈련

mglearn.discrete_scatter(X[:,0], X[:,1], kmeans.labels_, markers='o')

# 중심 표시
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0],
                         kmeans.cluster_centers_[:,1],[0,1,2], markers='^', markeredgewidth=2)
plt.show()



