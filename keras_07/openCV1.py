import matplotlib.pyplot as plt
import cv2

img = cv2.imread('test.jpg') # 그림파일을 BGR 형식 데이터로 
plt.axis('off') # 격자 제거
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # 컬러 형식을 RGR -> RGB 형변환
plt.show()
