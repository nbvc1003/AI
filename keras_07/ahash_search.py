from PIL import Image
import numpy as np
import os, re

## 이미지의 형태를 기준으로 이미지 비교하는 예제

search_dir = "./images/101_ObjectCategories"
cache_dir = "./images/cache_avhash"
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)  # 없으면 생성


# 이미지를 데이터 Average hash로 변환
def average_hash(fname, size=16):
    # fname 파일전체 +
    fname2 = fname[len(search_dir):]  # 파일명에서 경로 내용을 제거 -> 파일 이름만 추출
    cache_file = cache_dir + "/" + fname2.replace('/', '_') + ".csv"  # / -> _ 바꾸고 확장자 csv로
    if not os.path.exists(cache_file):
        img = Image.open(fname)
        img = img.convert('L').resize((size, size), Image.ANTIALIAS)
        pixels = np.array(img.getdata()).reshape(size, size)
        avg = pixels.mean()
        px = 1 * (pixels > avg)  # bool 값을 int 값으로
        np.savetxt(cache_file, px, fmt='%.0f', delimiter=',')  # csv형식으로 저장
    else:  # 이미 있으면 로드
        px = np.loadtxt(cache_file, delimiter=',')
    return px


# 해밍 거리 구하기
def hamming_dist(a, b):
    aa = a.reshape(1, -1)  # 1차원 으로 변경
    ab = b.reshape(1, -1)
    dist = (aa != ab).sum()  # 같지 않은 갯수합
    return dist


# 모든 폴더에 적용
def enum_all_files(path):
    for root, dirs, files in os.walk(path):  # path 의 루트, 디렉토리들, 파일들
        for f in files:
            # os.path.join
            fname = os.path.join(root, f)# 해당 파일의 전체 경로
            if re.search(r'\.(jpg|jpeg|png)$', fname): # 파일의 확장자가 .. 인지 여부
                yield fname  # 값을 반환하고 종료되지 않고 계속 실행..


# 이미지 찾기
def find_image(fname, rate):
    src = average_hash(fname)
    for fname in enum_all_files(search_dir):
        fname = fname.replace('\\', '/')
        dst = average_hash(fname)
        diff_r = hamming_dist(src, dst) / 256
        if diff_r < rate: # 차이값이 0.25 이하인경우 찾기 0 이면 같은 이미지
            yield (diff_r, fname)


# 찾기
srcfile = search_dir + "/chair/image_0016.jpg"
html = ""
# 0.25 : 75이상 픽셀이 동일한것을 찾아라
sim = list(find_image(srcfile, 0.25))  # 조건에 맞는 파일들의 거리값과 이름을 list로 출력
sim = sorted(sim, key=lambda x: x[0])  # 가까운 순서로 정렬
for r, f in sim:  # r은 유사도 , f파일명
    print(r, ">", f)
    s = '<div style="float:left;"><h3>[ 차이 :' + str(r) + '-' + \
        os.path.basename(f) + ']</h3>' + \
        '<p><a href="' + f + '"><img src="' + f + '" width=400>' + \
        '</a></p></div>'
    html += s
# html로 출력
html = """<html><head><meta charset="utf8"></head>
<body><h3>원래 이미지</h3><p>
<img src='{0}' width=400></p>{1}
</body></html>""".format(srcfile, html)

with open("./avhash-search-output.html", "w", encoding="utf-8") as f:
    f.write(html)

print("ok")
