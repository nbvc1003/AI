from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import mglearn
# 한글 사용
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1,2,figsize=(10, 3))
for model, ax in zip([LinearSVC(max_iter=5000, C=5), LogisticRegression(solver='liblinear')], axes):
    clf = model.fit(X, y) # 훈련

    # 2차원 평면 그래프              모델, 훈련데이터, 축, 투명도
    mglearn.plots.plot_2d_separator(clf, X, ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:,0],X[:,1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
axes[0].legend()
mglearn.plots.plot_linear_svc_regularization()
plt.show()

    



    