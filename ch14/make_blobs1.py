from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus']= False

plt.title("세개의 클러스터를 가진 가상의 데이터")
# 임의의 데이터 생성 .. random_state=1 : 1 랜덤seed를 유지한다.
X, y = make_blobs(n_features=2,  centers=3, random_state=1) # 독립변수 2, 중심점 3
plt.scatter(X[:,0], X[:,1], marker='o', c=y, s=100, edgecolors='k', linewidths=2)
plt.xlabel('특성0')
plt.ylabel('특성1')
plt.legend(['클래스0','클래스1','클래스2'])
plt.show()
