import urllib.request as req
import gzip, os, os.path

savepath = "./mnist" # . 현재 폴더 아래 폴더 생성..
baseurl = "http://yann.lecun.com/exdb/mnist"
files = ["train-images-idx3-ubyte.gz",
         "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz",
         "t10k-labels-idx1-ubyte.gz"]
if not os.path.exists(savepath): #디덱토리가 없으면 만들어라
    os.mkdir(savepath)

# 파일들을 다운로드 .
for f in files:
    url = baseurl + "/" + f
    loc = savepath + "/" + f
    print("download", url)
    if not os.path.exists(loc):
        req.urlretrieve(url, loc) # loc 패스에 파일 다운 로드..


for f in files:
    gz_file = savepath+"/"+f
    raw_file = savepath +"/" + f.replace(".gz", "")
    print("gzip :", f)
    with gzip.open(gz_file, 'rb') as fp: # byte 로 읽기
        body = fp.read()
        with open(raw_file, "wb") as w: 
            w.write(body) # 압축풀려서 저장









