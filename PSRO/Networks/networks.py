import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(in_dim, 64)
        # self.fc2 = nn.Linear(64, 64)
        self.mu_layer = nn.Linear(64, out_dim)
        self.log_std_layer = nn.Linear(64, out_dim)

    def forward(self, state):
        x = F.tanh(self.fc1(state))
        # x = F.tanh(self.fc2(x))
        mu = F.tanh(self.mu_layer(x))
        log_std = F.tanh(self.log_std_layer(x))
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        action = dist.sample()
        action = torch.clamp(action, min=-1, max=1)
        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(in_dim, 64)
        # self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, state):
        x = F.tanh(self.fc1(state))
        # x = F.tanh(self.fc2(x))
        value = self.out(x)

        return value
