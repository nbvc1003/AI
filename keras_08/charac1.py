import numpy as np
import string
samples = ['The cat sat on the mat', 'The dog ate my homework']

characters = string.printable # 영문자 대소 숫자, 특수문자..
print(characters)
token_index = dict(zip(characters, range(1,len(characters)+1))) # +1 은 0번 인덱스를 사용하지 않겠다는 의미

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.values())+1))
for i, sample in enumerate(samples):
    for j, character in enumerate(samples[:max_length]):
        index = token_index.get(character)
        results[i, j, index] = 1.
print(results)


