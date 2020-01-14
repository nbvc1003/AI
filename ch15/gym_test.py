import gym
from gym.envs.registration import  register
import msvcrt
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
# key 맵핑
arrow_keys = {
    75:LEFT,
    77:RIGHT,
    72:UP,
    80:DOWN
}

register(id='FrozenLake-v3', entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'map_name':'4x4',
                                                                                    'is_slippery':False})

env = gym.make('FrozenLake-v3')
env.render()
while True:

    # Choose an action from keyboard
    # print(ord(msvcrt.getch()))
    # 키입력시 값이 2번 들어온다.
    if ord(msvcrt.getch()) != 224:  # 특수키 입력은 224
        print("Game aborted!, not arrow")
        break
    key = ord(msvcrt.getch())

    if key not in arrow_keys.keys():
        print("게임종료", key)
        break
    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    print("state :", state, "action :", action,' reward :', reward, ' info:', info)

    if done:
        print('게임 종료 :', reward)
        break
    








