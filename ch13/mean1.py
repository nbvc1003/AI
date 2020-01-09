import pandas as pd
df = pd.read_csv('tem10y.csv', encoding='utf-8')

#날짜별 list
md = {} # 저장용
for i , row in df.iterrows(): # DataFrame을 한row씩 읽어옴
    m,d,v = (int(row['월']),int(row['일']), float(row['기온']))
    key = str(m)+"/"+str(d)  # 파이썬은 숫자와 문자의 + 연산이 안됨
    if not (key in md): # md 안에 key가 없다면
        md[key] = []
    md[key] += [v] # 해당 키에 list형식으로 데이터들을 채운다.
    # print(md)

# 날짜별 평균 구하기
avs = {} # 평균을 저장할 변수
for key in md.keys():
    v = avs[key] = sum(md[key])/len(md[key])
    print("{} : {}".format(key, v))
print(avs['1/9'])
