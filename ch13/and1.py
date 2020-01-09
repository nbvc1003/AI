import pandas as pd
from sklearn import svm, metrics

and_data = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
and_df = pd.DataFrame(and_data)
and_data = and_df[[0, 1]] # 0,1열  데이터
and_label = and_df[2]  #  2열

clf = svm.SVC()
clf.fit(and_data, and_label)
pre = clf.predict(and_data)
ac_score = metrics.accuracy_score(and_label, pre)
print("정확도 :", ac_score)