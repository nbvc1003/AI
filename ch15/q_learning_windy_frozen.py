import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random

register(id='FrozenLake-v3', entry_point='gym.envs.toy_text:FrozenLakeEnv')

env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n, env.action_space.n])
# learning_rate = 0.85
dis = 0.9

num_episodes = 2000
rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    while not done:
        action = np.argmax(Q[state, :]+ np.random.randn(1, env.action_space.n)/(i+1))
        new_state, reward, done, _ = env.step(action)
        Q[state, action] = reward + dis * np.max(Q[new_state, :])
        rAll += reward
        state = new_state
    rList.append(rAll)
print('성공율 ', sum(rList)/num_episodes)
print("왼, 밑, 우, 위")
print(Q)
plt.bar(range(len(rList)), rList, color='b')

plt.show()

