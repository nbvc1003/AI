import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

faces_all = fetch_olivetti_faces()

K = 20 # 20번호에 해당하는 사람 얼굴 사진
pca3 = PCA(n_components=2) # 주성분 2개 를 찾아낸다.
X3 = faces_all.data[faces_all.target==K]
W3 = pca3.fit_transform(X3) # 위분석 결과를 토대로 X3의 차원 축소
X32 = pca3.inverse_transform(W3) # 다시 차원 복귀 (결과적으로 주성분이 강조된 형태로)

# .reshape(64, 64)  : 원래 이미지 형태로 복원
face_mean = pca3.mean_.reshape(64, 64) # 평균 얼굴 이미지
face_p1 = pca3.components_[0].reshape(64, 64) # 첫번째 특성
face_p2 = pca3.components_[1].reshape(64, 64) # 두번째 특성

N = 2 # 2행
M = 5 # 5열
fig = plt.figure(figsize=(10,5))
# top에서 1픽셀 간격, bottom 간격 0 hspace=0, wspace=0.05 사진사이 간격
plt.subplots_adjust(top=1, bottom=0, hspace=0, wspace=0.05)
for i in range(N):
    for j in range(M):
        k = i *M +j
        ax = fig.add_subplot(N,M,k+1)
        w = 1.5*(k-5) if k < 5 else 1.5*(k-4)
        # 주성분을 반영해서 다양한 조합
        # face_p2 성분을 더하거나 마이너스 했을때 얼굴이 웃는 모습으로 변한다.
        # 따라서 face_p2성분은 웃는 얼굴 여부 판단성분..
        ax.imshow(face_mean+w*face_p2, cmap=plt.cm.bone)
        ax.grid(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.title("주성분 비중:{}".format(w))
plt.suptitle("평균얼굴에 주성분 2을 더한 사진")
plt.tight_layout()
plt.show()