import torch
import mazeworld3 as mazeworld
import my_drqn3 as my_drqn

from my_drqn3 import DRQN

env = mazeworld.gameEnv()
model = my_drqn.Model(env=env, config=my_drqn.config)

net_model=torch.load("model_drqn.pkl")
model.model = net_model


epsilon = 0.1

R = 0
s = env.reset()
while True:
    a = model.get_action(s, epsilon)
    print(s, a)
    s, reward, done = env.step(a)
    R += reward
    if done == True:
        break
    print(R)
