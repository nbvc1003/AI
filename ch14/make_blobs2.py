from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import mglearn
import numpy as np
from sklearn.svm import LinearSVC


from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus']= False

plt.title("세개의 클러스터를 가진 가상의 데이터")
# 임의의 데이터 생성 .. random_state=1 : 1 랜덤seed를 유지한다.
X, y = make_blobs(n_features=2,  centers=3, random_state=1) # 독립변수 2, 중심점 3
linear_svc = LinearSVC().fit(X,y)
print("계수 배열 ", linear_svc.coef_.shape) # [3, 2
print("절편 배열 ", linear_svc.intercept_.shape) # [3]
mglearn.plots.plot_2d_classification( linear_svc,X, fill=True, alpha=0.7)
mglearn.discrete_scatter(X[:,0],X[:,1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svc.coef_, linear_svc.intercept_, mglearn.cm3.colors):
    # y = -coef_0/ coef_1 * X - intercept /coef_1 특성 2개 구분선 식
    plt.plot(line, -(line*coef[0] + intercept)/coef[1])
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel('특성 0')
plt.ylabel('특성 1')
plt.legend(['클래스0','클래스1','클래스2','클래스0경계','클래스1경계','클래스2경계'], loc=(1.01, 0.3))
plt.show()

