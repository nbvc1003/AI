import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

faces_all = fetch_olivetti_faces()

K = 20 # 20번호에 해당하는 사람 얼굴 사진
pca3 = PCA(n_components=2) # 주성분 2개 분석 비지도 학습
X3 = faces_all.data[faces_all.target==K]
W3 = pca3.fit_transform(X3) # 위분석 결과를 토대로 X3의 차원 축소
X32 = pca3.inverse_transform(W3) # 다시 차원 복귀 (결과적으로 주성분이 강조된 형태로)

face_mean = pca3.mean_.reshape(64, 64) # 평균 얼굴 이미지
face_p1 = pca3.components_[0].reshape(64, 64)
face_p2 = pca3.components_[1].reshape(64, 64)
plt.subplot(131) # 1행 3열 1번째  (1,3,1)
plt.imshow(face_mean, cmap=plt.cm.bone) # 평균얼굴 기본색 (흑백)
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.title('평균얼굴')

plt.subplot(132)
plt.imshow(face_p1, cmap=plt.cm.bone)
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.title('주성분 1')

plt.subplot(133)
plt.imshow(face_p2, cmap=plt.cm.bone)
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.title('주성분 2')

plt.show()





