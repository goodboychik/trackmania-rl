# REDQ Implementation for Trackmania

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

# Actor Module (same as Baseline)
class REDQActorModule(TorchActorModule):
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

# Ensemble of Critic Networks
class REDQCritic(nn.Module):
    def __init__(self, observation_space, action_space, ensemble_size=10):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.q_networks = nn.ModuleList([BaselineCritic(observation_space, action_space)
                                         for _ in range(ensemble_size)])

    def forward(self, obs, act):
        q_values = torch.stack([q_net(obs, act) for q_net in self.q_networks], dim=0)
        return q_values  # Shape: [ensemble_size, batch_size]

# Actor-Critic Network
class REDQActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, ensemble_size=10):
        super().__init__()
        self.actor = REDQActorModule(observation_space, action_space)
        self.critic = REDQCritic(observation_space, action_space, ensemble_size)

# Training Agent
class REDQAgent(TrainingAgent):
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None,
                 action_space=None,
                 device=None,
                 ensemble_size=10,
                 gamma=0.99,
                 lr=1e-3,
                 utd_ratio=20):
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)
        self.model = REDQActorCritic(observation_space, action_space, ensemble_size).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.gamma = gamma
        self.lr = lr
        self.utd_ratio = utd_ratio
        self.ensemble_size = ensemble_size
        self.q_params = self.model.critic.parameters()
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):
        for _ in range(self.utd_ratio):
            o, a, r, o2, d, _ = batch
            pi, logp_pi = self.model.actor(obs=o, compute_logprob=True)
            q_values = self.model.critic(o, a)

            with torch.no_grad():
                a2, logp_a2 = self.model.actor(o2)
                next_q_values = self.target_model.critic(o2, a2)
                min_q, _ = torch.min(next_q_values, dim=0)
                backup = r + self.gamma * (1 - d) * (min_q - logp_a2)

            # Randomly sample M networks
            M = 2  # Subset size
            indices = np.random.choice(self.ensemble_size, M, replace=False)
            q_values_subset = q_values[indices]
            loss_q = ((q_values_subset - backup.unsqueeze(0)) ** 2).mean()

            self.q_optimizer.zero_grad()
            loss_q.backward()
            self.q_optimizer.step()

            # Update actor
            q_pi = self.model.critic(o, pi)
            min_q_pi, _ = torch.min(q_pi, dim=0)
            loss_pi = (-min_q_pi).mean()

            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            self.pi_optimizer.step()

            # Update target networks
            with torch.no_grad():
                for p, p_targ in zip(self.model.parameters(), self.target_model.parameters()):
                    p_targ.data.mul_(0.995)
                    p_targ.data.add_(0.005 * p.data)

        return {'loss_actor': loss_pi.item(), 'loss_critic': loss_q.item()}
