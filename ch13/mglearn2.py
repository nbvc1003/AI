import matplotlib.pyplot as plt
import mglearn

# 한글 사용
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# 랜덤하게 테스트용 데이터 생성..
X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:,0], X[:,1], y)
print(X[:,0],X[:,1],y)
plt.legend(['클래스 0','클래스 1'], loc=4)
print("X.shape :{}".format(X.shape))
plt.show()



