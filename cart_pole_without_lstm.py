import gym
import torch
import torch.nn as nn
from itertools import count
from torch.distributions import Bernoulli
import numpy as np
import torch.nn.functional as F
from tensorboardX import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        # self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x, hidden = self.lstm(x, hidden)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.sigmoid(self.fc3(x))
        return x

    def select_action(self, state):
        with torch.no_grad():
            prob = self.forward(state)
            b = Bernoulli(prob)
            action = b.sample()
        return action.item()


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2, 64)
        # self.lstm = nn.LSTM(64, 256, batch_first=True)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x, hidden = self.lstm(x, hidden)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    policy = PolicyNetwork().to(device)
    value = ValueNetwork().to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=1e-4)
    value_optim = torch.optim.Adam(value.parameters(), lr=3e-4)
    gamma = 0.99
    writer = SummaryWriter('./fc_logs')
    steps = 0

    for epoch in count():
        state = env.reset()
        state = np.delete(state, 1)
        state = np.delete(state, 2)
        episode_reward = 0

        rewards = []
        actions = []
        states = []

        for time_steps in range(200):
            states.append(state)
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            action = policy.select_action(state)
            actions.append(int(action))
            next_state, reward, done, _ = env.step(int(action))
            next_state = np.delete(next_state, 1)
            next_state = np.delete(next_state, 2)
            episode_reward += reward
            state = next_state
            rewards.append(reward)
            if done:
                break

        R = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            rewards[i] = R

        rewards_mean = np.mean(rewards)
        rewards_std = np.std(rewards)
        rewards = (rewards - rewards_mean) / rewards_std

        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.FloatTensor(actions).unsqueeze(1).to(device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)

        # print(batch_state.shape, batch_next_state.shape, batch_action.shape, batch_reward.shape)

        with torch.no_grad():
            v = value(states_tensor)
            advantage = rewards_tensor - v

        prob = policy(states_tensor)
        # print(prob.shape)
        b = Bernoulli(prob)
        log_prob = b.log_prob(actions_tensor)
        loss = - log_prob * advantage
        # print(loss.shape)
        loss = loss.mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        writer.add_scalar('action loss', loss.item(), epoch)

        v = value(states_tensor)
        value_loss = F.mse_loss(rewards_tensor, v)

        value_optim.zero_grad()
        value_loss.backward()
        value_optim.step()
        writer.add_scalar('value loss', value_loss.item(), epoch)

        writer.add_scalar('episode reward', episode_reward, epoch)
        if epoch % 10 == 0:
            print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))
            torch.save(policy.state_dict(), 'fc-policy.para')





