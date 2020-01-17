from keras.datasets import imdb
# imdb : 영화평가 글 db 5만건
# 영화평가글 중에서 자주 사용하는 단어 10000개까지만
(X_train, Y_train),(X_test,Y_tset) = imdb.load_data(num_words=10000)
print(X_train[0]) # 단어의 인덱스 0 ~ 9999
print(Y_train[0]) #  0: 부정평가, 1: 긍정평가
print([max(i) for i in X_train]) #   [max(i) for i in X_train] : 표현식 X_train 2차원 배열에서 각배열별 가장큰 값을 배열로
print(max([max(i) for i in X_train])) #

# 단어와 숫자를 맵핑한 사전
word_index = imdb.get_word_index()

# (key, value) -> (value, key) 변경
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#  padding, 문서시작, 사전에 없음 -> 앞에 3개의 공통 정보가 있음 따라서 3개를 걸러서 맴핑
decoded_review = " ".join([reverse_word_index.get(i-3, '?') for i in X_train[0]]) # get(i-3, '?') -> 값이 없으면 디폴트 ? 으로
print(decoded_review)


