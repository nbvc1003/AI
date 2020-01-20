from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split

## from keras.preprocessing.image import ImageDataGenerator 를 사용하는 방법과 다른
## 2번째 방법 ..

# 분류대상 카테고리
caltech_dir = "./images/101_ObjectCategories"
categories = ["chair","camera","butterfly","elephant","flamingo"]
nb_classes = len(categories)
# 이미지 크기 지정
image_w = 64
image_h = 64
pixels = image_h * image_w * 3 # 가로 세로 RGB
# 이미지 데이터
X = [] # 실제 이미지 데이터
Y = [] # 카테고리 항목

for idx , cat in enumerate(categories):
    label = [0 for i in range(nb_classes)]  # [ 0,0,0,0,.....] 전부 0으로 채워서
    label[idx] = 1
    imgage_dir = caltech_dir+"/"+cat
    # 위디덱토리에서 .jpg파일만 찾아서
    files = glob.glob(imgage_dir+"/*.jpg")
    for i,f in enumerate(files):
        img = Image.open(f) # 이미지 파일을 RGB 파일로
        img = img.convert('RGB') # 이미지 파일을 RGB 값으로
        img = img.resize((image_w, image_h))
        data = np.asarray(img) # img 데이터를 numpy 배열로
        X.append(data)
        Y.append(label) # 위에서 one_hot 형식으로 된 값을 넣는다.
        if i%10 == 0:
            print(i, '\n', data)
X = np.array(X)
Y = np.array(Y)
# 훈련 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./images/5obj.npy", xy)
print("ok,", len(Y))





