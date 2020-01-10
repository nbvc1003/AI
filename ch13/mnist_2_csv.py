import struct



def to_csv(name, maxdata):
    # 레이블과 이미지 파일 열기
    lbl_f = open("./mnist/" + name+"-labels-idx1-ubyte", "rb")
    img_f = open("./mnist/" + name + "-images-idx3-ubyte", "rb")
    csv_f = open("./mnist/"+name+".csv","wt", encoding='utf-8')
    
    # 파일시리얼번호, 데이터사이즈
    mag, lbl_count = struct.unpack(">ii",lbl_f.read(8)) # 8byte 읽음
    print(mag, lbl_count)
    mag, img_count = struct.unpack(">ii", img_f.read(8))

    rows, cols = struct.unpack(">ii", img_f.read(8)) # 행, 열의 갯수
    pixels = rows * cols
    res = []
    for idx in range(lbl_count):
        if idx > maxdata: # 데이터 일부 (maxdata만큼) 만 처리
            break
        label = struct.unpack("B", lbl_f.read(1))[0] # "B" : 부호가 없는 정수로 읽는다.
        bdata = img_f.read(pixels) # 픽셀개수 만큼
        sdata = list(map(lambda n : str(n) ,  bdata)) # 요소들을 문자로 바꿔서 다시 리스트로
        csv_f.write(str(label)+',')
        csv_f.write(",".join(sdata)+"\r\n")
    csv_f.close()
    lbl_f.close()
    img_f.close()

to_csv("train", 1000)
to_csv("t10k", 500)

    
    









