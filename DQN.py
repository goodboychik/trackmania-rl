# DQN with Discretized Actions for Trackmania

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tmrl.envs import TM2020Env
from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property

# Discretize actions
def discretize_actions(action_space, num_bins):
    action_bins = []
    for dim in range(action_space.shape[0]):
        bins = np.linspace(action_space.low[dim], action_space.high[dim], num_bins)
        action_bins.append(bins)
    return np.array(np.meshgrid(*action_bins)).T.reshape(-1, action_space.shape[0])

# Simple CNN for DQN
class DQNCNN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQNCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flat_features = self._get_flat_features()
        self.fc = nn.Linear(self.flat_features, 512)
        self.output_layer = nn.Linear(512, num_actions)

    def _get_flat_features(self):
        return 64 * 7 * 7  # Assuming input image size is 84x84

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.flat_features)
        x = F.relu(self.fc(x))
        q_values = self.output_layer(x)
        return q_values

# Training Agent for DQN
class DQNAgent(TrainingAgent):
    def __init__(self,
                 observation_space=None,
                 action_space=None,
                 device=None,
                 num_bins=5,
                 gamma=0.99,
                 lr=1e-3,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay=10000):
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)
        self.num_bins = num_bins
        self.actions = discretize_actions(action_space, num_bins)
        self.num_actions = len(self.actions)
        self.q_network = DQNCNN(input_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_network = deepcopy(self.q_network).to(self.device)
        self.optimizer = Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def act(self, obs, test=False):
        self.steps_done += 1
        sample = np.random.rand()
        epsilon_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
                            np.exp(-1. * self.steps_done / self.epsilon_decay)
        if sample > epsilon_threshold or test:
            with torch.no_grad():
                q_values = self.q_network(obs)
                action_index = q_values.max(1)[1].item()
        else:
            action_index = np.random.randint(self.num_actions)
        action = self.actions[action_index]
        return action

    def train(self, batch):
        o, a, r, o2, d, _ = batch
        q_values = self.q_network(o)
        action_indices = [np.where((self.actions == act).all(axis=1))[0][0] for act in a.cpu().numpy()]
        state_action_values = q_values.gather(1, torch.tensor(action_indices).unsqueeze(1).to(self.device))

        with torch.no_grad():
            next_q_values = self.target_network(o2)
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            expected_state_action_values = r + self.gamma * (1 - d) * max_next_q_values

        loss = F.mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
