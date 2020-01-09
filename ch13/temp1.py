
in_file = 'tempdata.csv'
out_file = 'tem10y.csv'

with open(in_file, "r", encoding='euc_kr') as fr: # defalut 'rt'
    lines = fr.readlines()

# 제목 컬럼을 넣고 6째출부터 추가.
lines = ["연,월,일,기온,품질,균질\n"] + lines[5:]
lines = map(lambda v:v.replace('/', ','), lines) # '/' -> ',' 바꾼다.
result = ''.join(lines).strip()
print(result)

with open(out_file, "w", encoding='utf-8') as fw: # defalut 'wt'
    fw.write(result)

print("저장 완료")