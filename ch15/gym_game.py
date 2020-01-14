import gym
env = gym.make('FrozenLake-v0')
observation = env.reset() # 초기화

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    # done종료, 끝에 도착, hole에 빠지거나
    # env.step: 실제 이동 가봐야 보상이나 상태를 알 수 있다.
    observation, reward, done, info = env.step(action)



