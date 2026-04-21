# 🎮 Pacman Reinforcement Learning Research Framework

A comprehensive, research-grade Reinforcement Learning experimentation framework built on the Berkeley Pacman environment. Implements and compares multiple RL algorithms — both flat and hierarchical — with full training pipelines, automatic logging, and performance plots.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Algorithms Implemented](#algorithms-implemented)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Guide](#training-guide)
- [Hierarchical RL Training](#hierarchical-rl-training)
- [Evaluation & Visualization](#evaluation--visualization)
- [Configuration](#configuration)
- [Results & Outputs](#results--outputs)
- [Experiment Reference](#experiment-reference)

---

## Project Overview

This framework extends the Berkeley Pacman codebase into a full RL research platform. It supports:

- **Flat RL Agents**: Tabular Q-Learning, SARSA, Approximate Q-Learning, DQN, REINFORCE, PPO
- **Hierarchical RL**: Meta-controller (DQN) + 4 specialized skill agents for Pacman
- **Multi-Agent**: RL-controlled Ghost agents with alternating training support
- **Shaped Rewards**: Dense, phase-specific reward functions for faster learning
- **Auto Logging**: JSON/CSV metrics + Matplotlib plots generated automatically

---

## Project Structure

```
ai_pacman/
├── README.md
└── src/
    ├── config.yaml              # Main hyperparameter configuration
    ├── test_config.yaml         # Quick smoke-test config (few episodes)
    ├── train.py                 # Flat RL training pipeline
    ├── hierarchical_train.py    # 4-phase hierarchical training pipeline
    ├── evaluate.py              # Agent evaluation with optional graphics
    │
    ├── agents/
    │   ├── q_agent.py           # Tabular Q-Learning & Approximate Q-Learning
    │   ├── sarsa_agent.py       # Tabular SARSA
    │   ├── dqn_agent.py         # Deep Q-Network (CNN + Experience Replay)
    │   ├── reinforce_agent.py   # Monte-Carlo Policy Gradient (REINFORCE)
    │   ├── ppo_agent.py         # Proximal Policy Optimization (PPO + GAE)
    │   ├── ghost_agent.py       # RL Ghost wrapper (Q-Learning based)
    │   ├── hierarchical_pacman.py  # Hierarchical Pacman (Meta + Skills)
    │   ├── hierarchical_ghost.py   # Hierarchical Ghost agent
    │   └── skills/
    │       ├── base_skill.py       # Abstract base for all skill agents
    │       ├── food_skill.py       # Skill: Navigate to nearest food
    │       ├── power_pellet_skill.py  # Skill: Navigate to power pellet
    │       ├── chase_skill.py      # Skill: Chase scared ghosts
    │       └── escape_skill.py     # Skill: Evade dangerous ghosts
    │
    ├── models/
    │   ├── dqn_net.py           # CNN Q-Network for DQN
    │   └── actor_critic_net.py  # Shared Actor-Critic backbone (PPO/REINFORCE)
    │
    ├── utils/
    │   ├── state_parser.py      # Game state → CNN tensor (6 channels)
    │   ├── replay_buffer.py     # Experience replay for off-policy methods
    │   ├── logger.py            # Metric logging + Matplotlib plots
    │   └── reward_shaper.py     # Shaped reward functions for hierarchical agents
    │
    ├── results/                 # Auto-created: saved models, logs, plots
    └── layouts/                 # Pacman map layouts
```

---

## Algorithms Implemented

### Flat RL Agents

| Agent | Key | Notes |
|---|---|---|
| Tabular Q-Learning | `q_learning` | Epislon-greedy, tabular Q-table |
| Approximate Q-Learning | `approx_q` | Linear function approximation |
| SARSA | `sarsa` | On-policy tabular TD |
| DQN | `dqn` | CNN, target network, experience replay |
| REINFORCE | `reinforce` | Monte-Carlo PG with baseline |
| PPO | `ppo` | Clipped objective + GAE |

### Hierarchical RL (Pacman)

**Meta-Controller** (DQN) selects one of 4 high-level goals every N steps:

| Goal | Skill Agent | Behaviour |
|---|---|---|
| `eat_food` | `FoodSkill` | Navigate to nearest food pellet |
| `eat_power_pellet` | `PowerPelletSkill` | Navigate to nearest capsule |
| `chase_ghost` | `ChaseSkill` | Chase the nearest scared ghost |
| `escape_ghost` | `EscapeSkill` | Move away from dangerous ghosts |

### Hierarchical Ghost Agent

Rule-based meta-goal selection (`chase_pacman` / `scatter` / `ambush`) with Q-learning movement controller.

---

## Installation

### Prerequisites

- Python 3.9+
- Conda environment with PyTorch, NumPy, Matplotlib, PyYAML

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd ai_pacman

# Create and activate the conda environment
conda create -n Env python=3.9
conda activate Env

# Install dependencies
pip install torch numpy matplotlib pyyaml
```

---

## Quick Start

### 1. Run a quick smoke-test (5 episodes, fast):

```bash
conda run -n Env python src/train.py --config src/test_config.yaml --agent q_learning
```

### 2. Train DQN for 2000 episodes (full config):

```bash
conda run -n Env python src/train.py --config src/config.yaml --agent dqn
```

### 3. Watch a trained agent play (with graphics):

```bash
conda run -n Env python src/evaluate.py \
  --config src/config.yaml \
  --agent dqn \
  --model_path src/results/dqn_final.pt \
  --episodes 5
```

---

## Training Guide

All flat RL agents use `train.py`. Run from the **project root** (`ai_pacman/`).

### Syntax

```bash
conda run -n Env python src/train.py \
  --config src/config.yaml \
  --agent <AGENT_KEY> \
  [--episodes N]
```

### Examples

```bash
# Tabular Q-Learning — 3000 episodes
conda run -n Env python src/train.py --config src/config.yaml --agent q_learning --episodes 3000

# SARSA — 2000 episodes
conda run -n Env python src/train.py --config src/config.yaml --agent sarsa --episodes 2000

# Approximate Q-Learning
conda run -n Env python src/train.py --config src/config.yaml --agent approx_q --episodes 2000

# Deep Q-Network (DQN)
conda run -n Env python src/train.py --config src/config.yaml --agent dqn --episodes 2000

# REINFORCE
conda run -n Env python src/train.py --config src/config.yaml --agent reinforce --episodes 2000

# PPO
conda run -n Env python src/train.py --config src/config.yaml --agent ppo --episodes 2000
```

> **Tip:** `--episodes` overrides the `num_episodes` in your config without editing the file.

---

## Hierarchical RL Training

Uses `hierarchical_train.py` which automatically runs the 4-phase staged training:

| Phase | What trains | What is frozen |
|---|---|---|
| 1 — Skill Training | Pacman skill agents | Ghosts (baseline) |
| 2 — Meta-Controller | Meta-controller DQN | Skills + Ghosts |
| 3 — Ghost Training | Ghost agents | Pacman |
| 4 — Alternating | Pacman ↔ Ghosts alternately | Each other |

### Run Hierarchical Training

```bash
# With rule-based ghosts (recommended for initial training)
conda run -n Env python src/hierarchical_train.py --config src/config.yaml

# With hierarchical RL ghost agents
conda run -n Env python src/hierarchical_train.py --config src/config.yaml --ghost_mode hierarchical
```

> The number of episodes per phase = `num_episodes / 4` from the config.

---

## Evaluation & Visualization

### Headless (fast, no window)

```bash
conda run -n Env python src/evaluate.py \
  --config src/config.yaml \
  --agent q_learning \
  --model_path src/results/q_learning_final.pt \
  --episodes 10 \
  --headless
```

### With Graphics (visual playback)

```bash
conda run -n Env python src/evaluate.py \
  --config src/config.yaml \
  --agent dqn \
  --model_path src/results/dqn_final.pt \
  --episodes 5
```

> A Pacman game window will open and autoplay the trained agent.

### All Agent Types

| Agent | Model path |
|---|---|
| `q_learning` | `results/q_learning_final.pt` |
| `dqn` | `results/dqn_final.pt` |
| `ppo` | `results/ppo_final.pt` |
| `reinforce` | `results/reinforce_final.pt` |

---

## Configuration

The main config is at `src/config.yaml`. Key sections:

```yaml
env:
  layout: "mediumClassic"   # Pacman layout (see src/layouts/)
  num_ghosts: 2             # Number of ghost agents
  timeout: 60

training:
  num_episodes: 2000        # Total training episodes
  save_interval: 100        # Save model & plots every N episodes
  results_dir: "./results"

hyperparameters:
  gamma: 0.99               # Discount factor
  learning_rate: 0.0005
  batch_size: 64
  buffer_capacity: 50000    # Replay buffer size (DQN)
  epsilon_start: 1.0        # Exploration (DQN/Q-Learning)
  epsilon_end: 0.05
  ppo_epochs: 4             # PPO update epochs
  ppo_clip: 0.2             # PPO clip parameter

multiagent:
  mode: "baseline"          # "baseline" (rule ghosts) or "alternating" (RL ghosts)
  alternating_interval: 100

hierarchical:
  goal_interval: 5          # Steps between meta-controller decisions
  skill_alpha: 0.3          # Skill Q-learning rate
  skill_epsilon: 0.15       # Skill exploration
```

### Available Layouts

Located in `src/layouts/`. Common options:

| Layout | Description |
|---|---|
| `smallClassic` | Small, fast — good for debugging |
| `mediumClassic` | Standard Pacman grid |
| `originalClassic` | Full-size classic layout |

---

## Results & Outputs

All outputs are saved automatically to `src/results/`:

```
results/
├── *.pt / *.pkl            # Saved model weights
├── logs/
│   ├── *_metrics.json      # Episode rewards, win rates, losses
│   └── *_metrics.csv       # Same data in CSV format
└── plots/
    ├── *_reward.png         # Episode reward curve (with moving average)
    ├── *_win_rate.png       # Win rate over training
    ├── *_loss.png           # Training loss (DQN/PPO)
    ├── hierarchical_goal_freq.png   # Goal selection frequency bar chart
    └── hierarchical_skill_usage.png # Skill execution step counts
```

---

## Experiment Reference

### Compare all flat agents

```bash
for AGENT in q_learning sarsa dqn reinforce ppo; do
  conda run -n Env python src/train.py --config src/config.yaml --agent $AGENT --episodes 2000
done
```

### Flat PPO vs Hierarchical

```bash
# Flat PPO
conda run -n Env python src/train.py --config src/config.yaml --agent ppo --episodes 2000

# Hierarchical (same total episodes split across 4 phases)
conda run -n Env python src/hierarchical_train.py --config src/config.yaml
```

### Multi-Agent: RL Ghosts

```bash
# Edit config.yaml: multiagent.mode = "alternating"
conda run -n Env python src/train.py --config src/config.yaml --agent dqn --episodes 2000
```

---

## Authors

Built as a 3-month research project on Reinforcement Learning using the Berkeley Pacman environment.
