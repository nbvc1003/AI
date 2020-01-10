from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
iris = load_iris()
N = 10 # 붓꽃 10송이
X = iris.data[:10, :2] # 데이터 10개, 열2개 꽃받침 길이 , 폭

plt.plot(X.T, 'o:')
plt.xticks(range(4), ['꽃받침길이', '꽃받침 폭'])
plt.xlim(-0.5, 2)
plt.ylim(2.5, 6)
plt.title("붗 꽃의 크기 특성")
plt.legend(['표본 {}'.format(i+1) for i in range(N)])
plt.show()