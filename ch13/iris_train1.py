from sklearn import svm, metrics
import random , re
csv = []
with open('iris.csv', 'r', encoding='utf-8') as fp:
    for line in fp:
        line = line.strip() # 공백 줄바꿈 제거
        cols = line.split(',') # 쉼표로 분리 해서 list로
        # print(type(cols))
        fn = lambda n : float(n) if re.match(r'^[0-9\.]+$', n) else n
        cols = list(map(fn, cols)) # fn 조건으로 cols 요소들을 개별처리하고 list형으로 묶어서 출력 
        # print(cols)
        csv.append(cols)

del csv[0] # header 를 제거

random.shuffle(csv) # 셔플
total_len = len(csv)
train_len = int(total_len*2/3) #데이터의 2/3
train_data = []
train_label = []
test_data = []
test_label = []
for i in range(total_len):
    data = csv[i][0:4] # 숫자 파트 : 숫자와 종목을 분리
    label = csv[i][4] # 종목
    if i < train_len:
        train_data.append(data)
        train_label.append(label)
    else:
        test_data.append(data)
        test_label.append(label)

clf = svm.SVC()
clf.fit(train_data, train_label) # 훈련
pre = clf.predict(test_data) # 테스트
ac_score = metrics.accuracy_score(test_label, pre) # 정확도
print('정답을 :', ac_score)


