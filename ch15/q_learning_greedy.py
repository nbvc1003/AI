import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(id='FrozenLake-v3', entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'map_name':'4x4',
                                                                                    'is_slippery':False})

env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n, env.action_space.n])

# 기본 Q 러닝 알고 리즘에
# 랜덤 변수와 거리에 따른 디스카운트 로직 추가.

dis = 0.9
num_episodes = 2000
rList = []

for i in range(num_episodes):
    e = 1./((i//100)+1)
    state = env.reset()
    rAll = 0
    done = False
    while not done:
        if np.random.rand(1) < e :
            action = env.action_space.sample()
        else :
            action = np.argmax(Q[state, :])

        new_state, reward, done, _ = env.step(action)
        Q[state, action] = reward + dis * np.max(Q[new_state,:])
        rAll += reward
        state = new_state
    rList.append(rAll)
print('성공율 ', sum(rList)/num_episodes)
print("왼, 밑, 우, 위")
print(Q)
plt.bar(range(len(rList)), rList, color='b')

plt.show()




