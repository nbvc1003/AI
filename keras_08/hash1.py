import numpy as np
samples = ['The cat sat on the mat', 'The dog ate my homework']
dimensionality = 1000
max_length = 10
# 단어 1000개만 사용
results = np.zeros((len(samples), max_length, dimensionality))  #

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        # 단어를 해싱하여 0과 1,000 사이의 랜덤한 정수 인덱   스로  변환합니다.
        index = abs(hash(word)) % dimensionality
        results[i,j,index] = 1
print(results)