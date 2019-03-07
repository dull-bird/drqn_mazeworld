import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import random

import math

from utils.hyperparameters import Config
from agents.BaseAgent import BaseAgent
import env.mazeworld_basic as mazeworld

config = Config()

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#epsilon variables
config.epsilon_start = 0.2
config.epsilon_final = 0.01
config.epsilon_decay = 30000
config.epsilon_by_frame = lambda frame_idx: config.epsilon_final + (config.epsilon_start - config.epsilon_final) * math.exp(-1. * frame_idx / config.epsilon_decay)

#misc agent variables
config.GAMMA = 1
config.LR = 0.005

#memory
config.TARGET_NET_UPDATE_FREQ = 100
config.EXP_REPLAY_SIZE = 2000
config.BATCH_SIZE = 32

#Learning control variables
config.LEARN_START = 100
config.MAX_FRAMES = 50000
config.n_hidden1 = 20
config.n_hidden2 = 50

#define the deep-q-network
class DQN(nn.Module):
    def __init__(self, num_features, num_actions):
        super(DQN, self).__init__()

        self.num_features = num_features
        self.num_hidden1 = config.n_hidden1
        self.num_hidden2 = config.n_hidden2
        self.num_actions = num_actions

        #self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        #self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.hidden1 = nn.Linear(self.num_features, self.num_hidden1)
        self.hidden2 = nn.Linear(self.num_hidden1, self.num_hidden2)
        self.actions = nn.Linear(self.num_hidden2, self.num_actions)
        self.init()

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.actions(x)

        return x

    def init(self):
        self.hidden1.weight.data.normal_(0, 0.1)
        self.hidden2.weight.data.normal_(0, 0.1)
        self.actions.weight.data.normal_(0, 0.1)

class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Model(BaseAgent):
    def __init__(self, static_policy=False, env=None, config=None):
        super(Model, self).__init__()
        self.device = config.device

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START

        self.static_policy = static_policy
        self.num_feats = env.state_dim
        self.num_actions = env.action_dim
        self.env = env

        self.declare_networks()

        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        # move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

    def declare_networks(self):
        self.model = DQN(self.num_feats, self.num_actions)
        self.target_model = DQN(self.num_feats, self.num_actions)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)

    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,) + (self.num_feats,)

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        #print(batch_state)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        #print(batch_action)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device,
                                      dtype=torch.uint8)
        try:  # sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device,
                                                 dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = batch_vars

        # estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)

        # target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + (self.gamma * max_next_q_values)

        diff = (expected_q_values - current_q_values)
        # loss = self.huber(diff)
        loss = diff**2
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, frame=0):
        if self.static_policy:
            return None

        self.append_to_replay(s, a, r, s_)

        if frame < self.learn_start:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        self.save_loss(loss.item())
        self.save_sigma_param_magnitudes()

    def get_action(self, s, eps=0.1):
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def huber(self, x):
        cond = (x.abs() < 1.0).to(torch.float)
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

if __name__ == "__main__":
    env = mazeworld.gameEnv()


    model = Model(env = env, config=config)


    episode_reward = 0

    episode_num  = 0

    observation = env.reset()
    for frame_idx in range(1, config.MAX_FRAMES):
        epsilon = config.epsilon_by_frame(frame_idx)

        action = model.get_action(observation, epsilon)
        prev_observation = observation
        observation, reward, done = env.step(action)

        observation = None if done else observation

        #print(prev_observation, action, reward, observation, frame_idx)

        model.update(prev_observation, action, reward, observation, frame_idx)

        episode_reward += reward

        if done or episode_reward < -3000:
            print("episode", episode_num, episode_reward)
            observation = env.reset()
            model.save_reward(episode_reward)
            episode_reward = 0
            episode_num += 1

            if np.mean(model.rewards[-20:]) > 50:
                break


    print(model.rewards)

    #print(dqn.choose_action([0,0]))
    m = []
    for i in range(6):
        m.append([])
        for j in range(6):
            if env.maze[i, j] != 0:
                action = model.get_action([i, j], 0)
                if action == 0:
                    m[i].append("U")
                elif action == 1:
                    m[i].append("D")
                elif action == 2:
                    m[i].append("L")
                else:
                    m[i].append("R")
            else:
                m[i].append("-")

    m[2][5] = "X"
    print(m[0])
    print(m[1])
    print(m[2])
    print(m[3])
    print(m[4])
    print(m[5])