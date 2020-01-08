import tensorflow as tf
import numpy as np

# 하나의 문자열..
sentence = ("if you want to build a ship, don't drum up people together "
            "to collect wood and don't assign them tasks and work, but rather" 
            "teach them to long for the endless immensity of the sea.")
char_set = list(set(sentence)) # 중복된 철자를 제거
char_dic = {w:i for i, w in enumerate(char_set)} # 문자 리스트를 딕셔너리로
dataX = []
dataY = []
data_dim = len(char_set) #
hidden_size = len(char_set)
num_classes = len(char_set)
seq_length = 10 # input 단어를 문자 10개씩


for i in range(0, len(sentence) - seq_length):
    # 실제 x_data, y_data 의 갯수
    x_str = sentence[i:i + seq_length]
    y_str = sentence[i+1 : i+seq_length + 1]
    x = [char_dic[c] for c in x_str] # 문자열을 숫자로
    y = [char_dic[c] for c in y_str]
    print(i, x, y)
    dataX.append(x)
    dataY.append(y)








