import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

iris = load_iris()

N = 10 # 
# 행 10송이, 꽃받침 길이 , 폭
X = iris.data[:N, :2]
pca1 = PCA(n_components=1)
X_low = pca1.fit_transform(X) # 차원 축소
X2 = pca1.inverse_transform(X_low) # 차원 복귀

plt.figure(figsize=(7,7))
ax = sns.scatterplot(0,1, data= pd.DataFrame(X), s=100, color=".2", markers='s')
# 각각의 marker 에 표시
for i in range(N):
    d = 0.03 if X[i,1] > X[i,1] else -0.04
    ax.text(X[i,0]-0.065, X[i,1] + d, '표본{}'.format(i+1))
    # 투영되는 상황을 시작적으로 표현
    plt.plot([X[i,0], X2[i,0]], [X[i,1], X2[i,1]], 'k--')

# 차원이 축소된 투영된 위치표시
plt.plot(X2[:,0], X2[:,1], 'o-', markersize=10 )
# 평균값 포이트
plt.plot(X[:,0].mean(), X[:,1].mean(), markersize=10, marker='D')

# 십자 표시 ..
plt.axvline(X[:,0].mean(), c='r') # 수직
plt.axhline(X[:,1].mean(), c='r') # 수평

plt.xlabel("꽃받침의 길이")
plt.ylabel("꽃받침의 폭")
plt.title('iris 데이터의 1차원 축소')
plt.axis('equal')
plt.show()
# 평균 값..
print('mean :', pca1.mean_)
# 주성분의 수치
print('component :', pca1.components_)





