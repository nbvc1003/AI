from PIL import Image
import numpy as np
def average_hash(fname, size=16):
    img = Image.open(fname) # 이미지 open
    img = img.convert('L') # 이미지를 회색으로
    # 이미지 크리 변경
    img = img.resize((size,size), Image.ANTIALIAS) # Image.ANTIALIAS 경계선을 날카롭게
    pixel_data = img.getdata() # 그림을 픽셀데이터로
    pixels = np.array(pixel_data)
    pixels = pixels.reshape((size,size)) # 배열을 2차원으로
    avg = pixels.mean() # 이미지 전체의 색값의 평균을 구한다.
    diff = 1 * (pixels > avg) # 색의 값이 평균보다 크면 1 작으면 0
    return diff

ahash = average_hash('tower.jpg')
print(ahash)