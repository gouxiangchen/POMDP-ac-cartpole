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
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        x = self.relu(self.fc1(x))
        x, hidden = self.lstm(x, hidden)
        x = self.relu(x)
        x = self.sigmoid(self.fc2(x))
        return x, hidden

    def select_action(self, state, hidden):
        with torch.no_grad():
            prob, hidden = self.forward(state, hidden)
            b = Bernoulli(prob)
            action = b.sample()
        return action.item(), hidden


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2, 64)
        self.lstm = nn.LSTM(64, 256, batch_first=True)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x, hidden):
        x = self.relu(self.fc1(x))
        x, hidden = self.lstm(x, hidden)
        x = self.relu(x)
        x = self.fc2(x)
        return x, hidden


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    policy = PolicyNetwork().to(device)
    value = ValueNetwork().to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=1e-4)
    value_optim = torch.optim.Adam(value.parameters(), lr=3e-4)
    gamma = 0.99
    writer = SummaryWriter('./lstm_logs')
    steps = 0

    for epoch in count():
        state = env.reset()
        state = np.delete(state, 1)
        state = np.delete(state, 2)
        episode_reward = 0

        a_hx = torch.zeros((1, 1, 128)).to(device)
        a_cx = torch.zeros((1, 1, 128)).to(device)

        rewards = []
        actions = []
        states = []

        for time_steps in range(200):
            states.append(state)
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            action, (a_hx, a_cx) = policy.select_action(state, (a_hx, a_cx))
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

        states_tensor = torch.FloatTensor(states).unsqueeze(0).to(device)
        actions_tensor = torch.FloatTensor(actions).unsqueeze(1).to(device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)

        # print(batch_state.shape, batch_next_state.shape, batch_action.shape, batch_reward.shape)

        with torch.no_grad():
            c_hx = torch.zeros((1, 1, 256)).to(device)
            c_cx = torch.zeros((1, 1, 256)).to(device)
            v, v_hidden = value(states_tensor, (c_hx, c_cx))
            v = v.squeeze(0)
            advantage = rewards_tensor - v

        a_hx = torch.zeros((1, 1, 128)).to(device)
        a_cx = torch.zeros((1, 1, 128)).to(device)

        prob, a_hidden = policy(states_tensor, (a_hx, a_cx))
        prob = prob.squeeze(0)
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

        c_hx = torch.zeros((1, 1, 256)).to(device)
        c_cx = torch.zeros((1, 1, 256)).to(device)
        v, v_hidden = value(states_tensor, (c_hx, c_cx))
        v = v.squeeze(0)
        value_loss = F.mse_loss(rewards_tensor, v)
        value_optim.zero_grad()
        value_loss.backward()
        value_optim.step()
        writer.add_scalar('value loss', value_loss.item(), epoch)

        writer.add_scalar('episode reward', episode_reward, epoch)
        if epoch % 10 == 0:
            print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))
            torch.save(policy.state_dict(), 'lstm-policy.para')





