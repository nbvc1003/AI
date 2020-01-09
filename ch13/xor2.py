import pandas as pd
from sklearn import svm, metrics

xor_data = [[0,0,0],[0,1,1], [1,0,1], [1,1,0]]
xor_df = pd.DataFrame(xor_data)
xor_data = xor_df[[0,1]] # 0,1열  데이터
xor_label = xor_df[2]  #  2열

clf = svm.SVC()
clf.fit(xor_data, xor_label)
pre = clf.predict(xor_data)
ac_score = metrics.accuracy_score(xor_label, pre)
print("정확도 :", ac_score)