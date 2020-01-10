from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager, rc
import seaborn as sns
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
iris = load_iris()
N = 10 # 붓꽃 10송이
X = iris.data[:10, :2] # 데이터 10개, 열2개 꽃받침 길이 , 폭

plt.figure(figsize=(8,8))
#                X데이터,Y데이터  s : 사이즈,
ax = sns.scatterplot(0,1,data= pd.DataFrame(X), s=100, color='.2', markers='s')
for i in range(N):
    ax.text(X[i,0] - 0.05, X[i,1]+ 0.03, "표본 {}".format(i+1))
plt.xlabel("꽃 받침길이")
plt.ylabel("꽃 받침 폭")
plt.title("붗꽃 크기 특성(2차원)")
plt.axis('equal')
plt.show()