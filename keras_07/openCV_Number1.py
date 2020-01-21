import sys, cv2
import numpy as np

# 이미지 읽기
im = cv2.imread('./NumImgs/numbers.png')
# 회색
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

#  경계를 이진값으로 변경   blockSize: 11 , c:2
# 배경과 목적 이미지를 구분하기 위함
tresh = cv2.adaptiveThreshold(blur, 255, 1,1,11,2)

# 윤곽 추출
# cv2.RETR_EXTERNAL : 가장 바깥 라인만 추출
contours, hierachy = cv2.findContours(tresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if h < 20:
        continue # 너무작으면 스킵
    red = (0,0,255)
    # im 에 빨간색 사작형 추가.
    cv2.rectangle(im,(x,y),(x+w,y+h), red, 2)
cv2.imwrite('./NumImgs/numbers_cnt.png', im)
