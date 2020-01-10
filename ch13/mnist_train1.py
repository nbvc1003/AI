from sklearn import model_selection, svm, metrics

def load_csv(fname):
    labels = []
    images = []
    with open(fname, 'r') as f:
        for line in f:
            cols = line.split(',') #데이터를 , 로 분리해서 list로
            if len(cols) < 2 :
                continue
            labels.append(int(cols.pop(0)))
            vals = list(map(lambda n : int(n)/256, cols)) # 컬러값 조절
            images.append(vals)
        return {'labels':labels, 'images':images}
data = load_csv("./mnist/train.csv") # train 데이터
test = load_csv("./mnist/t10k.csv")
clf = svm.SVC()
clf.fit(data['images'], data['labels'])
predict = clf.predict(test['images'])
ac_score = metrics.accuracy_score(test['labels'], predict)
cl_report = metrics.classification_report(test['labels'], predict)
print('정답율 :', ac_score)
print('보고서 \n', cl_report)