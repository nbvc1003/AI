import matplotlib.pyplot as plt
import mglearn

# 한글 사용
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# 좌표의 마이너스 영역 표시 여부
plt.rcParams['axes.unicode_minus'] = False

# 테스트 데이터 생성
X, y = mglearn.datasets.make_wave(n_samples=40)

# 선형회귀 그래프
mglearn.plots.plot_linear_regression_wave()

plt.plot(X, y, 'o')
plt.xlabel('특성')
plt.ylabel('타겟')

plt.show()