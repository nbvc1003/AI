import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus']= False

iris = load_iris()
# OneVsOneClassifier 비해 속도는 빠르고 정확도는 떨어진다.
model_ovr =OneVsRestClassifier(LogisticRegression(solver='lbfgs')).fit(iris.data, iris.target)
ax1 = plt.subplot(211)
pd.DataFrame(model_ovr.decision_function(iris.data)).plot(ax=ax1, legend=True)
plt.title("판별함수")
ax2 = plt.subplot(212)
pd.DataFrame(model_ovr.predict(iris.data), columns=["prediction"]).plot(marker='o', ls='',ax=ax2)
plt.title('클래스 판별')
plt.tight_layout()
plt.show()
