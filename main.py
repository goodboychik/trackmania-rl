# Comprehensive Soft Actor-Critic (SAC) Reinforcement Learning Implementation
# This script implements a sophisticated deep reinforcement learning approach
# for training an agent in a custom environment using the SAC algorithm.

# Import necessary libraries for deep learning, environment interaction,
# and distributed training
import itertools
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from tmrl import GenericGymEnv
from torch.optim import Adam
from torch.distributions.normal import Normal
import json

# Import configuration and utility modules
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.util import partial
from tmrl.networking import Trainer, RolloutWorker, Server
from tmrl.training_offline import TrainingOffline
from tmrl.actor import TorchActorModule
from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property

# Import custom JSON encoding/decoding utilities
from q import TorchJSONEncoder, TorchJSONDecoder


# Custom Environment Class
# Extends GenericGymEnv to provide a specialized reward structure
# and environment interaction mechanism
class MyCustomTM2020Env(GenericGymEnv):
    def __init__(self, *args, **kwargs):
        # Initialize the base environment (using Pendulum-v1 as a template)
        super(MyCustomTM2020Env, self).__init__(id='Pendulum-v1', *args, **kwargs)
        # Track previous progress to calculate incremental rewards
        self.prev_progress = None

    def _reward(self, info):
        # Complex reward function that encourages progress, speed, and penalizes mistakes
        # Calculates rewards based on multiple environment attributes

        # Get current progress from environment info
        current_progress = info.get('distance', 0.0)

        # Initialize previous progress on first call
        if self.prev_progress is None:
            self.prev_progress = current_progress

        # Calculate progress delta to reward incremental improvement
        delta_progress = current_progress - self.prev_progress
        self.prev_progress = current_progress

        # Base reward proportional to progress
        reward = delta_progress * 10.0

        # Additional reward for speed to encourage faster movement
        speed = info.get('speed', 0.0)
        reward += speed * 0.1

        # Significant penalties for collisions and going off track
        collision = info.get('collision', False)
        if collision:
            reward -= 10.0
        off_track = info.get('off_track', False)
        if off_track:
            reward -= 10.0

        return reward

    def reset(self):
        # Reset progress tracking when environment is reset
        self.prev_progress = None
        return super(MyCustomTM2020Env, self).reset()


# Environment and Training Configuration
# Load configuration parameters for distributed machine learning setup
env_cls = MyCustomTM2020Env

# Extract training hyperparameters from configuration
epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]

# Device configuration for distributed training
device_trainer = 'cuda' if cfg.CUDA_TRAINING else 'cpu'
device_worker = 'cpu'

# Memory and batch configuration
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]
batch_size = 256

# Network configuration parameters
server_ip_for_trainer = cfg.SERVER_IP_FOR_TRAINER
server_ip_for_worker = cfg.SERVER_IP_FOR_WORKER
server_port = cfg.PORT
password = cfg.PASSWORD
security = cfg.SECURITY

# Image and observation preprocessing parameters
window_width = cfg.WINDOW_WIDTH
window_height = cfg.WINDOW_HEIGHT
img_width = cfg.IMG_WIDTH
img_height = cfg.IMG_HEIGHT
img_grayscale = cfg.GRAYSCALE
imgs_buf_len = cfg.IMG_HIST_LEN
act_buf_len = cfg.ACT_BUF_LEN

# Utility function to create a configurable memory buffer
memory_base_cls = cfg_obj.MEM
memory_cls = partial(memory_base_cls,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=imgs_buf_len,
                     act_buf_len=act_buf_len,
                     crc_debug=False)

# Additional preprocessing configurations
sample_compressor = cfg_obj.SAMPLE_COMPRESSOR
sample_preprocessor = None
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR

# Hyperparameters for policy distribution log standard deviation
LOG_STD_MAX = 2
LOG_STD_MIN = -20


# Neural Network Utility Functions
# Create a multi-layer perceptron (MLP) with configurable layers and activations
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# Calculate the total number of features in a flattened tensor
def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


# Calculate output dimensions of a 2D convolution layer
def conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) /
                  conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) /
                  conv_layer.stride[1] + 1)
    return h_out, w_out


# Advanced Convolutional Neural Network for State Representation
# Combines image processing with additional state information
class ImprovedCNN(nn.Module):
    def __init__(self, q_net):
        super(ImprovedCNN, self).__init__()

        # Determine if this is a Q-network or policy network
        self.q_net = q_net

        # Initial image dimensions
        self.h_out, self.w_out = img_height, img_width

        # Convolutional layers for image feature extraction
        # Progressively reduce spatial dimensions while increasing feature depth
        self.conv1 = nn.Conv2d(imgs_buf_len, 64, kernel_size=5, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels

        # Calculate total flattened features after convolutions
        self.flat_features = self.out_channels * self.h_out * self.w_out

        # Prepare input features for final MLP
        float_features = 12 if self.q_net else 9
        self.mlp_input_features = self.flat_features + float_features

        # Configure MLP layers based on network type
        self.mlp_layers = [512, 512, 1] if self.q_net else [512, 512]
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)

    def forward(self, x):
        # Unpack input based on network type (Q-network or policy network)
        if self.q_net:
            speed, gear, rpm, images, act1, act2, act = x
        else:
            speed, gear, rpm, images, act1, act2 = x

        # Apply convolutional layers with batch normalization and ReLU activation
        x = F.relu(nn.BatchNorm2d(64)(self.conv1(images)))
        x = F.relu(nn.BatchNorm2d(128)(self.conv2(x)))
        x = F.relu(nn.BatchNorm2d(256)(self.conv3(x)))
        x = F.relu(nn.BatchNorm2d(256)(self.conv4(x)))

        # Flatten convolutional features
        flat_features = num_flat_features(x)
        x = x.view(-1, flat_features)

        # Concatenate image features with additional state information
        if self.q_net:
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)
        else:
            x = torch.cat((speed, gear, rpm, x, act1, act2), -1)

        # Pass through final MLP
        x = self.mlp(x)
        return x


# Actor Module: Policy Network for Action Generation
# Responsible for generating actions based on current state observations
class MyActorModule(TorchActorModule):
    def __init__(self, observation_space, action_space):
        # Initialize the actor module with observation and action space specifications
        super().__init__(observation_space, action_space)

        # Extract action dimension and action limits from action space
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

        # Use the improved CNN as the base network for feature extraction
        self.net = ImprovedCNN(q_net=False)

        # Create layers for mean (mu) and log standard deviation of action distribution
        # These layers will parameterize a Gaussian policy
        self.mu_layer = nn.Linear(512, dim_act)  # Mean of the action distribution
        self.log_std_layer = nn.Linear(512, dim_act)  # Log standard deviation
        self.act_limit = act_limit

    def save(self, path):
        # Custom save method using a JSON encoder to handle torch tensors
        with open(path, 'w') as json_file:
            json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)

    def load(self, path, device):
        # Custom load method using a JSON decoder to handle torch tensors
        self.device = device
        with open(path, 'r') as json_file:
            state_dict = json.load(json_file, cls=TorchJSONDecoder)
        self.load_state_dict(state_dict)
        self.to_device(device)
        return self

    def forward(self, obs, test=False, compute_logprob=True):
        # Forward pass to generate actions with probabilistic sampling

        # Extract features from observations using the CNN
        net_out = self.net(obs)

        # Generate mean and log standard deviation for action distribution
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)

        # Clamp log standard deviation to prevent extreme values
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Create a Normal distribution for action sampling
        pi_distribution = Normal(mu, std)

        # Determine action selection strategy
        if test:
            # During testing, use deterministic mean action
            pi_action = mu
        else:
            # During training, sample from the distribution using reparameterization trick
            pi_action = pi_distribution.rsample()

        # Compute log probabilities if required
        if compute_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # Additional correction for the tanh squashing transformation
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        # Apply tanh to squash actions to [-1, 1] range and scale by action limit
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        pi_action = pi_action.squeeze()

        return pi_action, logp_pi

    def act(self, obs, test=False):
        # Simplified method for action selection during inference
        with torch.no_grad():
            a, _ = self.forward(obs=obs, test=test, compute_logprob=False)
            return a.cpu().numpy()


# Q-Value Function for Critic Network
# Estimates the value of state-action pairs
class ImprovedCNNQFunction(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        # Use the improved CNN with Q-network configuration
        self.net = ImprovedCNN(q_net=True)

    def forward(self, obs, act):
        # Combine observations and actions for Q-value estimation
        x = (*obs, act)
        q = self.net(x)
        return torch.squeeze(q, -1)


# Actor-Critic Model Combining Policy and Value Estimation
class ImprovedActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        # Create an actor (policy) and two Q-critics
        self.actor = MyActorModule(observation_space, action_space)
        self.q1 = ImprovedCNNQFunction(observation_space, action_space)
        self.q2 = ImprovedCNNQFunction(observation_space, action_space)


# Soft Actor-Critic Training Agent
# Implements the learning algorithm for the actor-critic model
class SACTrainingAgent(TrainingAgent):
    # Cached property to create a no-gradient copy of the model
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None,
                 action_space=None,
                 device=None,
                 model_cls=ImprovedActorCritic,
                 gamma=0.99,
                 polyak=0.995,
                 alpha=0.2,
                 lr_actor=1e-4,
                 lr_critic=1e-4):
        # Initialize the training agent with configurable hyperparameters
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)

        # Create the actor-critic model and move to specified device
        model = model_cls(observation_space, action_space)
        self.model = model.to(self.device)

        # Create a target network for stable learning
        self.model_target = no_grad(deepcopy(self.model))

        # SAC-specific hyperparameters
        self.gamma = gamma  # Discount factor
        self.polyak = polyak  # Exponential moving average rate for target network
        self.alpha = alpha  # Entropy regularization coefficient
        self.lr_actor = lr_actor  # Learning rate for policy network
        self.lr_critic = lr_critic  # Learning rate for Q-networks

        # Combine parameters of both Q-networks
        self.q_params = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())

        # Create optimizers for actor and critic
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(self.q_params, lr=self.lr_critic)

        # Convert alpha to a tensor for computational efficiency
        self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        # Return a no-gradient version of the actor for inference
        return self.model_nograd.actor

    def train(self, batch):
        # Main training method implementing Soft Actor-Critic update

        # Unpack the training batch
        o, a, r, o2, d, _ = batch

        # Generate actions and log probabilities for the current observations
        pi, logp_pi = self.model.actor(obs=o, test=False, compute_logprob=True)

        # Compute Q-values for the current state-action pairs
        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)

        with torch.no_grad():
            # Generate actions for next states
            a2, logp_a2 = self.model.actor(o2)

            # Compute target Q-values using the target network
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            # Compute the backup value (target) using the Bellman equation
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha_t * logp_a2)

        # Compute critic (Q-network) loss
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Update Q-networks
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network parameters for policy update
        for p in self.q_params:
            p.requires_grad = False

        # Compute policy loss
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (self.alpha_t * logp_pi - q_pi).mean()

        # Update policy network
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network parameters
        for p in self.q_params:
            p.requires_grad = True

        # Soft update of target network
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # Return loss information for logging
        ret_dict = dict(
            loss_actor=loss_pi.detach().item(),
            loss_critic=loss_q.detach().item(),
        )
        return ret_dict

# Partial function to create a training agent with specific configurations
training_agent_cls = partial(SACTrainingAgent,
                             model_cls=ImprovedActorCritic,
                             gamma=0.99,
                             polyak=0.995,
                             alpha=0.2,
                             lr_actor=1e-4,
                             lr_critic=1e-4)

# Partial function to create the training process with specified parameters
training_cls = partial(
    TrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=epochs,
    rounds=rounds,
    steps=steps,
    update_buffer_interval=update_buffer_interval,
    update_model_interval=update_model_interval,
    max_training_steps_per_env_step=max_training_steps_per_env_step,
    start_training=start_training,
    device=device_trainer)


# Custom observation preprocessor to prepare input for the neural network
def improved_obs_preprocessor(obs):
    # Extract and normalize various observation components
    speed = np.array([0.0 if len(obs) < 1 else obs[0]], dtype=np.float32)
    gear = np.array([0.0 if len(obs) < 2 else obs[1]], dtype=np.float32)
    rpm = np.array([0.0 if len(obs) < 3 else obs[2]], dtype=np.float32)
    images = [] if len(obs) < 1 else obs[0]  # Assume images are already a NumPy array
    act1 = np.array([0.0 if len(obs) < 4 else obs[3]], dtype=np.float32)
    act2 = np.array([0.0 if len(obs) < 5 else obs[4]], dtype=np.float32)
    return speed, gear, rpm, images, act1, act2


# Main execution block for distributed machine learning setup
if __name__ == "__main__":
    from argparse import ArgumentParser

    # Parse command-line arguments to determine the role of this script
    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true', help='launches the server')
    parser.add_argument('--trainer', action='store_true', help='launches the trainer')
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')
    parser.add_argument('--test', action='store_true', help='launches a rollout worker in standalone mode')
    args = parser.parse_args()

    # Server initialization
    if args.server:
        import time
        serv = Server(port=server_port,
                      password=password,
                      security=security)
        while True:
            time.sleep(1.0)

    # Trainer initialization
    elif args.trainer:
        my_trainer = Trainer(training_cls=training_cls,
                             server_ip=server_ip_for_trainer,
                             server_port=server_port,
                             password=password,
                             security=security)
        my_trainer.run()

    # Rollout worker initialization
    elif args.worker or args.test:
        rw = RolloutWorker(env_cls=env_cls,
                           actor_module_cls=MyActorModule,
                           sample_compressor=sample_compressor,
                           device=device_worker,
                           server_ip=server_ip_for_worker,
                           server_port=server_port,
                           password=password,
                           security=security,
                           max_samples_per_episode=cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"],
                           obs_preprocessor=improved_obs_preprocessor,
                           standalone=args.test)
        rw.run(test_episode_interval=10)
    elif args.server:
        import time
        serv = Server(port=server_port,
                      password=password,
                      security=security)
        while True:
            time.sleep(1.0)
