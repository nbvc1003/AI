import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random

env = gym.make('FrozenLake-v0') # 미끄러지는 버전

Q = np.zeros([env.observation_space.n, env.action_space.n]) # 환경수, 행동수
learning_rate = 0.8 # 80% 확률로 새로움 값을 받아들임
dis = 0.9 # 거리에 따른 디스 카운트 비율

num_episodes = 2000
rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    while not done:
        action = np.argmax(Q[state, :]+ np.random.randn(1, env.action_space.n)/(i+1))
        new_state, reward, done, _ = env.step(action)
        # Q의 값을 현재값과 다음Q로부터의 값을 2:8 비율로 셋팅
        Q[state, action] = (1-learning_rate) * Q[state, action] + \
                           learning_rate * (reward + dis * np.max(Q[new_state, :]))
        rAll += reward
        state = new_state
    rList.append(rAll)
print('성공율 ', sum(rList)/num_episodes)
print("왼, 밑, 우, 위")
print(Q)
plt.bar(range(len(rList)), rList, color='b')

plt.show()

