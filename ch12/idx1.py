sample = "if you want you"


#   set (문자열)  문자열에서 중복되지 않는 글자들의 배열을 출력
idx2char = list(set(sample))
print(idx2char)
char2idx = {c: i for i,c in enumerate(idx2char)}
print(char2idx)
sample_idx = [char2idx[c] for c in sample]
print(sample_idx)
x_data = [sample_idx[:-1]] #"if you want yo"
print(x_data)
y_data = [sample_idx[1:]] # "f you want you"
print(y_data)

