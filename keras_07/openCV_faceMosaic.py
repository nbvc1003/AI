import cv2, sys, re

# 원본 위치
image_file = "./fac1_out.png"
# 표현식 설명 : 문자열끝이 조건 확장자로 끝나면 문자열뒤에 추가
out_file = re.sub(r'\.jpg|jpeg|png$', '_mosaic.jpg', image_file)
mosaic_rate = 30
# 케스케이드 경로
cascade_file = "./photo/haarcascade_frontalface_alt.xml"
# 이미지 읽기
img = cv2.imread(image_file)
image_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 그레이 스케일로
# 얼굴인식
cascade = cv2.CascadeClassifier(cascade_file)
face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=1, minSize=(100,100))
if len(face_list) == 0:
    print("사람 얼굴이 없습니다.")
    quit()
print(face_list) # x,y , w, h 값출력
color = (0,0,255)
for (x,y,w,h) in face_list:
    # 얼굴 잘라 오기
    face_img = img[y:y+h, x:x+w]
    # 얼굴 확대/ 축소 하기
    face_img = cv2.resize(face_img, (w//mosaic_rate, h//mosaic_rate))
    # 비율로 원래 대로
    face_img = cv2.resize(face_img, (w,h),interpolation=cv2.INTER_AREA)
    #원위치에 붙이기
    img[y:y+h, x:x+w] = face_img

# 렌더링 결과를 파일에 출력
cv2.imwrite(out_file, img)







