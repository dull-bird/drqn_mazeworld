import torch
import mazeworld3 as mazeworld
import my_drqn_I as my_drqn

from my_drqn3 import DRQN

env = mazeworld.gameEnv()
model = my_drqn.Model(env=env, config=my_drqn.config)

net_model=torch.load("model_drqn_I.pkl")
model.model = net_model


epsilon = 0.1

R = 0
s = env.reset()
a = 0
while True:
    a = model.get_action([s[0], s[1], a], epsilon)
    print(s, a)
    s, reward, done = env.step(a)
    R += reward
    if done == True:
        break
    print(R)
