import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random

def rargmax(vector):
    m = np.amax(vector) # vector 의 최대값 반환
    idices = np.nonzero(vector == m)[0] # 최대값인 것만 골라서
    return random.choice(idices) # 그중 하나를 랜덤하게 반환

register(id='FrozenLake-v3', entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'map_name':'4x4',
                                                                                    'is_slippery':False})

env = gym.make('FrozenLake-v3')
Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000 # 반복횟수
rList = []   # 보상에 해당하는 값
for i in range(num_episodes):
    state = env.reset() # 상태초기화
    rAll = 0 # 보상은 초기에 전부 0
    done = False # 종료여부 hole에 빠지면 종료
    while not done:  # 종료전에는 계속
        action = rargmax(Q[state, :]) # state 의 값중에 가장큰값 을 action에
        new_state , reward, done,_ = env.step(action)
        Q[state, action] = reward + np.max(Q[new_state,:])
        rAll += reward
        state = new_state
    rList.append(rAll)
print("성공율 :",  sum(rList) / num_episodes)
print("최종 Q table 값")
print("왼, 밑, 우, 위")
print(Q.shape)
print(Q)

plt.bar(range(len(rList)), rList, color='blue')
plt.show()






