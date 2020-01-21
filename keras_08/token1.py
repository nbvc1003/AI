from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat', 'The dog ate my homework']

# 가장 많이 사용하는 단어 1000개 선정해서 tokenizer
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)

# 문장이 단어들의 인덱스 리스트로 변경
sequences = tokenizer.texts_to_sequences(samples)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

print(sequences)
print(one_hot_results)

word_index = tokenizer.word_index
print('고유한 단어 갯수 ', word_index)