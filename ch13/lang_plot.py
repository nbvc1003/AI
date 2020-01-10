import matplotlib.pyplot as plt
import pandas as pd
import json
with open("./lang/freq.json", 'r',encoding='utf-8') as fp:
    freq = json.load(fp)

# print(type(freq), len(freq))
# print(freq[0])
# print(freq[1])

lang_dic = {}
# 0 : train, 1 : test
for i , lbl in enumerate(freq[0]['labels']):
    fq = freq[0]['freqs'][i]
    if not (lbl in lang_dic):
        lang_dic[lbl] = fq
        continue
    for idx, v in enumerate(fq):
        print(idx, v)
        lang_dic[lbl][i] = (lang_dic[lbl][idx]+v)/2

# 97 = 'a' , 98 -> 'b' ascii코드를 문자로 변경
asclist = [[chr(n) for n in range(97, 97+26)]]
df = pd.DataFrame(lang_dic, index=asclist)
plt.style.use('ggplot')
df.plot(kind='bar', subplots=True)
# df.plot(kind='line')

plt.show()


