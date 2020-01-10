from sklearn import svm, metrics

import glob, os.path, re, json



def check_freq(fname):
    name = os.path.basename((fname))
    lang = re.match(r'^[a-z]{2,}',name).group() # 파일명 앞에 두글자만으로 그룹핑
    with open(fname, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.lower() # 문장을 소문자로 변경
    #
    cnt = [0 for n in range(0,26)]
    code_a = ord("a") # ascii 값으로 변경
    code_z = ord("z")
    # 알파벳 출현 횟수
    for ch in text:
        n = ord(ch)
        if code_a <= n <= code_z: # 소문자 a~z 인결우만 카운트
            cnt[n-code_a] += 1 # 해당 문자가 들어 올때 마다 카운트 증가

    tot = sum(cnt) # 전체 문자수..
    # 갯수를 비율로 만들어 준다.
    freq = list(map(lambda n:n/tot, cnt)) # 각요소를 tot로 나눈다.
    return (freq, lang)

def load_files(path):
    freqs = []
    labels = []

    # 해당디덱토리의 파일 목록 출력
    file_list = glob.glob(path)
    for fname in file_list:
        f, l = check_freq(fname)
        freqs.append(f)
        labels.append(l)
    return {"freqs":freqs, "labels":labels}

data = load_files("./lang/train/*.txt")
test = load_files("./lang/test/*.txt")
with open("./lang/freq.json", 'w', encoding='utf-8') as fp:
    json.dump([data, test], fp)

clf = svm.SVC()
clf.fit(data['freqs'], data['labels'])
predict = clf.predict(test['freqs'])
ac_score = metrics.accuracy_score(test['labels'], predict)
cl_report = metrics.classification_report(test['labels'], predict)
print('정답율 :', ac_score)
print('보고서 \n', cl_report)








    





                    

