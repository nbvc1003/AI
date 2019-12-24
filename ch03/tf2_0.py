import tensorflow as tf
import numpy as np

x_train = np.arange(10) # 0~9
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0,  6.3,6.6, 7.4 ,8.0 , 9.0])

class TFLinereg(object): # object 모든 객체의 조상 생략 가능
    def __init__(self, learning_rate= 0.01):

        # 초기 사용할 변수와 함수를 선언
        self.w = tf.Variable(tf.zeros(shape=(1)))  # 초기값 0
        self.b = tf.Variable(tf.zeros(shape=(1)))  # 초기값 0
        # tf.keras.optimizers.SGD 경사하강법함수
        self.optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    # 훈련시킨다는 의미 fit
    def fit(self, x, y, num_epochs=10):
        # num_epochs : 한번에 묶어서 연산하는 단위
        training_costs = []
        for step in range(num_epochs):
            
            with tf.GradientTape() as tape: # 기록을 남긴다.
                hypothesis = self.w * x + self.b # 함수 # hypothesis : 가설
                mean_cost = tf.reduce_mean(tf.square(hypothesis - y)) #

            grads = tape.gradient(mean_cost, [self.w, self.b])
            self.optimizer.apply_gradients(zip(grads, [self.w,self.b])) # zip 묶어주는 기능.
            ## y = [1,2,3,4], y = [5,6,7,8]
            # zip(x,y) -> [[1,5],[2,6],[3,7],[4,8]]
            # numpy() 값을 numpy()  이용하여 숫자로 변경..
            training_costs.append(mean_cost.numpy()) #
        return training_costs

    def predict(self, x):
        return self.w * x + self.b

model = TFLinereg()
training_costs = model.fit(x_train, y_train)
import matplotlib.pyplot as plt
plt.plot(range(1, len(training_costs)+1), training_costs)  # 훈련횟수와 비용의관계 그래프
plt.tight_layout()
plt.xlabel('Train Epoch')
plt.ylabel('cost')
plt.show()
plt.scatter(x_train, y_train, marker='s', s=50, label="Train Data")
plt.plot(range(x_train.shape[0]), model.predict(x_train), color='red', marker='o',markersize=6, linewidth=3
         , label="Line")

plt.xlabel('x')
plt.legend()
plt.show()







