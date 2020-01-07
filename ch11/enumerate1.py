

a1 = ['body','bar','air']
# enumerate 값과 인덱스를 동시에 출력
for i , name in enumerate(a1):
    print(i, name)

names = ['철수','영희','길동']
for i, name in enumerate(names):
    print(i, name)

sample = 'if you want you'
# print(type(set(sample)))
idx2char = list(set(sample)) # se() 단어별로 자른다. 중복된 단어를 제거된
# 숫자를 문자로 변경, 또는 문자를 숫자로 변경..
char2idx = {c: i for i, c in enumerate(idx2char)} # {문자: 인덱스} 형식의 딕셔너리로 바꾼다.
print(char2idx)


