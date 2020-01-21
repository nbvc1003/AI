import sys, cv2
import numpy as np
import keras_07.openCV_Number3 as mnis1

# 훈련데이터 읽기

mnist = mnis1.build_model()
mnist.load_weights('mnist.hdf5')
# 이미지 읽기
im = cv2.imread('./NumImgs/numbers100.png')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
tresh = cv2.adaptiveThreshold(blur, 255,1,1,11,2)
cv2.imwrite('./numbers100_th.png', tresh)
coutours, hireachy = cv2.findContours(tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

# 숫자를 추출하여 좌표 정렬하기
rects = []
im_w = im.shape[1] # 이미지의 폭 크기
for i,cnt in enumerate(coutours):
    x,y,w,h = cv2.boundingRect(cnt)
    if w < 10 or h < 10 : 
        continue
    if w > im_w / 5: # 너무 커도 제외
        continue
    y2 = round(y/10)*10 # 한자리수제거 10단위로 -> index 값을 정의하기 위하여
    index = y2 * im_w + x
    rects.append((index, x,y,w,h))
# key=lambda x:x[0]  x[0] 값을 기준으로
rects = sorted(rects, key=lambda x:x[0]) # 인덱스 순으로 정렬
# 결과적으로 각 글자들의 영역정보들에 인덱스를 부여 한 배열 생성
print(rects)
X = []
for i , r in enumerate(rects):
    index, x,y,w,h = r
    num = gray[y:y+h, x:x+w] # 이미지 하나 추출
    num = 255 - num # 반전
    ww = round((w if w > h else h)*1.85) # w or h 큰쪽 값으로 선택
    spc = np.zeros((ww,ww)) #
    wy = (ww - h) // 2
    wx = (ww - w) // 2 # 채워주는 기준 위치를 찾기 위한
    spc[wy:wy+h, wx:wx+w] = num # 원래 이미지 사이즈 보다 더큰영역을 확보하고 그곳에 img 값을 채운다.

    ##  결과적으로 이미지 사이즈를 1.85배 확대하고 가로세로 비율을 1:1로 조정한다.
    num = cv2.resize(spc, (28,28)) # mnist 와 글자크기 맞추기
    num = num.reshape(28 * 28)
    num = num.astype("float32") / 255
    X.append(num)
# 예측
s = "31415926535897932384" + \
    "62643383279502884197" + \
    "16939937510582097494" + \
    "45923078164062862089" + \
    "98628034825342117067"
answer = list(s)
ok = 0
# 손글씨로 훈현한 모델로 그림에 있는 글자를 예측
nlist = mnist.predict(np.array(X))
for i, n in enumerate(nlist):
    ans = n.argmax() # one_hot -> int
    if ans == int(answer[i]):
        ok +=1
    else:
        print('에러 ', i,' 번째 ', ans, answer[i])
print('정답율 :', ok/len(nlist))









    

