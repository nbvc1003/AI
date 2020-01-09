from sklearn import svm
xor_data = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]] # p, q, result
data = []
label = []
# data 와 label만들기
for row in xor_data:
    p = row[0]
    q = row[1]
    r = row[2]
    data.append([p,q])
    label.append(r)

clf = svm.SVC() # SVC 모델생성
clf.fit(data, label) # 훈련
pre = clf.predict(data) # 예측
# 결과 확인
ok = 0
total = 0
for idx , answer in enumerate(label):
    p = pre[idx]
    if p == answer:
        ok += 1
    total += 1
print("정답률 :{}/{}= {}".format(ok, total,  ok/total*100))










