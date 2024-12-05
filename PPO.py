# PPO Implementation for Trackmania

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tmrl.envs import TM2020Env
from tmrl.training import TrainingAgent
from torch.distributions import Normal

# Recurrent Network for PPO
class RecurrentActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.gru = nn.GRU(input_size=observation_space.shape[0], hidden_size=256, num_layers=1)
        self.fc_actor = nn.Linear(256, action_space.shape[0])
        self.fc_critic = nn.Linear(256, 1)

    def forward(self, obs, hx):
        x, hx = self.gru(obs.unsqueeze(0), hx)
        x = x.squeeze(0)
        action_mean = self.fc_actor(x)
        value = self.fc_critic(x)
        return action_mean, value, hx

# Training Agent for PPO
class PPOAgent(TrainingAgent):
    def __init__(self,
                 observation_space=None,
                 action_space=None,
                 device=None,
                 gamma=0.99,
                 lr=3e-4,
                 clip_param=0.2,
                 ppo_epochs=10,
                 mini_batch_size=64):
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)
        self.model = RecurrentActorCritic(observation_space, action_space).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

    def act(self, obs, hx, test=False):
        with torch.no_grad():
            action_mean, value, hx = self.model(obs.to(self.device), hx)
            if test:
                action = action_mean
            else:
                dist = Normal(action_mean, torch.ones_like(action_mean) * 0.1)
                action = dist.sample()
            action = torch.tanh(action)
        return action.cpu().numpy(), value.cpu().numpy(), hx

    def train(self, rollouts):
        obs, actions, log_probs, returns, advantages, hx = rollouts

        for _ in range(self.ppo_epochs):
            for idx in BatchSampler(SubsetRandomSampler(range(len(advantages))), self.mini_batch_size, False):
                action_mean, values, _ = self.model(obs[idx], hx[:, idx, :])
                dist = Normal(action_mean, torch.ones_like(action_mean) * 0.1)
                new_log_probs = dist.log_prob(actions[idx]).sum(-1)
                ratio = torch.exp(new_log_probs - log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages[idx]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), returns[idx])
                loss = action_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return {'loss': loss.item()}
