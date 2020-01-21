import cv2
import sys

# 입력 파일

image_file = "./photo/face2.jpg"
# 케스케이드 정보
casecade_file = "./photo/haarcascade_frontalface_alt.xml"
#이미지 읽기
image = cv2.imread(image_file)
image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 얼굴인식 파일 읽기
casecade = cv2.CascadeClassifier(casecade_file)
# scaleFactor=1.1 다양한 크기로 비교할때 원 이미지 사이즈에 scaleFactor 값의 비율로 크기를 줄여가면서 비교
face_list = casecade.detectMultiScale(image_gs, scaleFactor=1.01, minNeighbors=1, minSize=(150,150))
if len(face_list) > 0 :
    print(face_list)
    color = (0,0,255) # BGR 빨강색
    for face in face_list:
        x,y,w,h = face
        # 찾은 얼굴위치에 빨강 사각형 그리기
        cv2.rectangle(image,(x,y), (x+w,y+h), color, thickness=8 )
    #파일로 출력
    cv2.imwrite("fac1_out.png",image)
else:
    print("그림이 없다 ")