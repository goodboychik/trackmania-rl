# Soft Actor-Critic (SAC) Reinforcement Learning Implementation for Trackmania

## Project Overview

This project implements a sophisticated Soft Actor-Critic (SAC) Reinforcement Learning framework designed for training autonomous agents in complex navigation environments. The implementation focuses on developing an intelligent agent capable of making optimal decisions in dynamic and challenging scenarios, like Trackmania game.

### Key Features
- Distributed machine learning architecture
- Advanced Soft Actor-Critic (SAC) algorithm implementation
- Customizable environment with complex reward mechanisms
- Deep neural network-based state representation
- Flexible configuration management

## Technical Architecture

### Core Components
- **Environment**: Custom environment extending GenericGymEnv
- **Neural Networks**:
  - Improved Convolutional Neural Network (CNN) for state representation
  - Actor-Critic network architecture
  - Advanced feature extraction from both visual and numeric inputs
- **Learning Algorithm**: Soft Actor-Critic (SAC)
  - Supports continuous action spaces
  - Learns optimal policies through value function approximation
  - Handles exploration-exploitation trade-offs

## Prerequisites

- Windows / Linux
- Python 3.8+
- PyTorch
- NumPy
- A recent NVIDIA GPU

## Installation Steps
**TrackMania 2020**

Install the game via its [official website](https://www.trackmania.com).

**Openplanet**

Required for the Gymnasium environment to compute rewards. Intall via its [official website](https://openplanet.dev/).

**TMRL Installation**:

- Run in a terminal:
    ```bash
    pip install tmrl
    ```
- During installation, accept the virtual gamepad driver license and install it.

## Running the Project

The script supports multiple execution modes:

1. Server Mode:
```bash
python main.py --server
```

2. Trainer Mode:
```bash
python main.py --trainer
```

3. Worker Mode:
```bash
python main.py --worker
```

4. Standalone Test Mode:
```bash
python main.py --test
```

# **For full pipeline, firstly run server, then run trainer and then connect as many workers as you have.**

## Potential Applications

- Autonomous driving simulations
- Robotic navigation
- Game AI development
- Adaptive control systems