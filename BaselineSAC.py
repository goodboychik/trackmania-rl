# Baseline SAC Implementation for Trackmania

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy
import itertools
from tmrl.envs import TM2020Env
from tmrl.actor import TorchActorModule
from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property

# Simple CNN for Baseline
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flat_features = self._get_flat_features()
        self.fc = nn.Linear(self.flat_features, 512)
        self.output_layer = nn.Linear(512, output_size)

    def _get_flat_features(self):
        return 64 * 7 * 7  # Assuming input image size is 84x84

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.flat_features)
        x = F.relu(self.fc(x))
        x = self.output_layer(x)
        return x

# Actor Module
class BaselineActorModule(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.net = SimpleCNN(input_channels=4, output_size=action_space.shape[0])
        self.act_limit = action_space.high[0]

    def forward(self, obs, test=False, compute_logprob=True):
        mu = self.net(obs)
        pi_action = torch.tanh(mu) * self.act_limit
        return pi_action, None

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs=obs, test=test, compute_logprob=False)
            return a.cpu().numpy()

# Critic Network
class BaselineCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = SimpleCNN(input_channels=4, output_size=1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        q_value = self.net(x)
        return q_value.squeeze(-1)

# Actor-Critic Network
class BaselineActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.actor = BaselineActorModule(observation_space, action_space)
        self.critic1 = BaselineCritic(observation_space, action_space)
        self.critic2 = BaselineCritic(observation_space, action_space)

# Training Agent
class BaselineSACAgent(TrainingAgent):
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None,
                 action_space=None,
                 device=None,
                 model_cls=BaselineActorCritic,
                 gamma=0.99,
                 polyak=0.995,
                 alpha=0.2,
                 lr=1e-3):
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)
        self.model = model_cls(observation_space, action_space).to(self.device)
        self.model_target = deepcopy(self.model).to(self.device)
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr = lr
        self.q_params = itertools.chain(self.model.critic1.parameters(), self.model.critic2.parameters())
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):
        o, a, r, o2, d, _ = batch
        pi, logp_pi = self.model.actor(obs=o, compute_logprob=True)
        q1 = self.model.critic1(o, a)
        q2 = self.model.critic2(o, a)

        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2)
            q1_pi_targ = self.model_target.critic1(o2, a2)
            q2_pi_targ = self.model_target.critic2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        for p in self.q_params:
            p.requires_grad = False

        q1_pi = self.model.critic1(o, pi)
        q2_pi = self.model.critic2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        for p in self.q_params:
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return {'loss_actor': loss_pi.item(), 'loss_critic': loss_q.item()}
