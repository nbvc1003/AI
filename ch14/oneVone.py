import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus']= False

iris = load_iris()
# 각특성을 1:1 로 비교
model_ovo = OneVsOneClassifier(LogisticRegression(solver='lbfgs')).fit(iris.data, iris.target)
print(model_ovo.decision_function(iris.data)) # 구분함수
ax1 = plt.subplot(211) # [2,1] 형태의 창에 1번째
#
pd.DataFrame(model_ovo.decision_function(iris.data)).plot(ax=ax1, legend=True)
plt.title('판별함수')
ax2 = plt.subplot(212)
# 훈련 결과에 실데이터를 적용하여 판정
pd.DataFrame(model_ovo.predict(iris.data), columns=["prediction"]).plot(marker='o',ls='',ax=ax2)
plt.title("클래스 판별")
plt.tight_layout()
plt.show()




