import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

faces_all = fetch_olivetti_faces()

K = 20 # 20번호에 해당하는 사람 얼굴 사진

faces = faces_all.images[faces_all.target==K]

N = 2 # 2행
M = 5 # 5열
fig = plt.figure(figsize=(10,5))
# top에서 1픽셀 간격, bottom 간격 0 hspace=0, wspace=0.05 사진사이 간격
plt.subplots_adjust(top=1, bottom=0, hspace=0, wspace=0.05)
for i in range(N):
    for j in range(M):
        k = i * M + j
        ax = fig.add_subplot(N,M, k+1)
        ax.imshow(faces[k], cmap=plt.cm.bone)
        ax.grid(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
plt.suptitle("올리베티 얼굴 사진")
plt.tight_layout()
plt.show()






