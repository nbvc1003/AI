import numpy as np
samples = ['The cat sat on the mat', 'The dog ate my homework']
# 데이터에 있는 모든 토큰에 인덱스 구축 
token_index = {}
for sample in samples:
    # split() 문장 의 단어별로 list 형으로 만듦
    for word in sample.split():
        if word not in token_index:
            # 단어 : index 형의 딕셔너리타입으로
            # +1 하면 단어 인덱스의 시작을 1번
            token_index[word] = len(token_index) + 1
print(token_index)

# 각단어를 one_hot 형식으로 바꿔 준다.
max_length = 10
result = np.zeros((len(samples), max_length, max(token_index.values())+1))

for i , sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        result[i,j,index] = 1
print(result)


