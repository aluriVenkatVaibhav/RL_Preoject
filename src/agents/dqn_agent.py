import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game import Agent, Directions
from utils.state_parser import StateParser
from utils.replay_buffer import ReplayBuffer
from models.dqn_net import DQNNet


class DQNAgent(Agent):
    """
    DQN Agent with the following improvements over the baseline:
      - Double DQN: policy net selects action, target net evaluates it
        → reduces Q-value overestimation bias
      - Soft (Polyak) target network update every step with tau=0.005
        → smoother, more stable target tracking
      - Warmup period: no gradient updates until buffer has enough samples
        → avoids learning from a non-representative buffer
      - Proper epsilon decay schedule driven by config (epsilon_decay_steps)
      - Gradient norm clipping via clip_grad_norm_ (cleaner than per-param clamp)
      - Dueling DQN network (handled in DQNNet)
    """

    def __init__(self, **kwargs):
        super(DQNAgent, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Hyperparameters ────────────────────────────────────────────────────
        self.gamma          = kwargs.get('gamma', 0.99)
        self.batch_size     = kwargs.get('batch_size', 64)
        self.lr             = kwargs.get('learning_rate', 1e-4)
        self.reward_scale   = kwargs.get('reward_scale', 1.0)
        self.reward_clip    = kwargs.get('reward_clip', None)

        # Soft target update (Polyak)
        self.tau                    = kwargs.get('target_update_tau', 0.005)
        self.target_update_interval = kwargs.get('target_update_interval', 1)

        # Epsilon-greedy exploration
        self.epsilon        = kwargs.get('epsilon_start', 1.0)
        self.epsilon_min    = kwargs.get('epsilon_end', 0.05)
        decay_steps         = kwargs.get('epsilon_decay_steps', 50000)
        self.epsilon_decay  = (self.epsilon - self.epsilon_min) / max(1, decay_steps)

        # Warmup: don't learn until buffer has this many transitions
        self.warmup_steps   = kwargs.get('warmup_steps', 1000)

        # Replay buffer
        buffer_capacity = kwargs.get('buffer_capacity', 50000)
        self.memory = ReplayBuffer(buffer_capacity)

        self.state_parser = StateParser()

        # Action mappings
        self.action_to_idx = {
            Directions.NORTH: 0,
            Directions.SOUTH: 1,
            Directions.EAST:  2,
            Directions.WEST:  3,
            Directions.STOP:  4,
        }
        self.idx_to_action = {v: k for k, v in self.action_to_idx.items()}
        self.num_actions = len(self.action_to_idx)
        self.avoid_stop = kwargs.get('avoid_stop', True)

        # Networks (initialised lazily on first state)
        self.policy_net = None
        self.target_net = None
        self.optimizer  = None

        self.steps_done = 0
        self.is_eval    = False

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path):
        if self.policy_net is not None:
            torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        import os
        if os.path.exists(path):
            self.load_path = path

    # ── Network initialisation ─────────────────────────────────────────────────

    def init_networks(self, state):
        if self.policy_net is None:
            self.state_parser.update_dims(state)
            input_shape = self.state_parser.input_shape

            self.policy_net = DQNNet(input_shape, self.num_actions).to(self.device)
            self.target_net = DQNNet(input_shape, self.num_actions).to(self.device)

            if hasattr(self, 'load_path') and self.load_path:
                self.policy_net.load_state_dict(
                    torch.load(self.load_path, map_location=self.device)
                )
                self.load_path = None

            # Sync target net weights
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    # ── Action selection ───────────────────────────────────────────────────────

    def getAction(self, state):
        self.init_networks(state)

        legal_actions = self._filtered_legal_actions(state)
        if not legal_actions:
            return None

        legal_idx = [self.action_to_idx[a] for a in legal_actions]

        # ε-greedy exploration
        if not self.is_eval and random.random() < self.epsilon:
            return random.choice(legal_actions)

        # Exploitation — mask illegal actions
        state_tensor = self.state_parser.get_tensor(state, self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0)

        masked_q = torch.full((self.num_actions,), -float('inf'), device=self.device)
        for idx in legal_idx:
            masked_q[idx] = q_values[idx]

        best_idx = masked_q.argmax().item()
        return self.idx_to_action[best_idx]

    def _filtered_legal_actions(self, state):
        legal_actions = state.getLegalActions(self.index)
        if self.avoid_stop and len(legal_actions) > 1 and Directions.STOP in legal_actions:
            return [action for action in legal_actions if action != Directions.STOP]
        return legal_actions

    def _legal_action_mask(self, state):
        mask = torch.zeros(self.num_actions, dtype=torch.bool)
        if state.isWin() or state.isLose():
            return mask

        legal_actions = self._filtered_legal_actions(state)
        for action in legal_actions:
            mask[self.action_to_idx[action]] = True
        return mask

    def _scale_reward(self, reward):
        reward = float(reward) * self.reward_scale
        if self.reward_clip is not None:
            reward = max(-self.reward_clip, min(self.reward_clip, reward))
        return reward

    # ── Learning step ──────────────────────────────────────────────────────────

    def update(self, state, action, nextState, reward, done):
        if self.is_eval:
            return None

        self.init_networks(state)

        # Store transition (keep tensors on CPU in buffer to save GPU memory)
        state_tensor      = self.state_parser.get_tensor(state,     'cpu').squeeze(0)
        next_state_tensor = self.state_parser.get_tensor(nextState,  'cpu').squeeze(0)
        action_idx        = self.action_to_idx[action]
        reward            = self._scale_reward(reward)
        next_legal_mask   = self._legal_action_mask(nextState)

        self.memory.push(state_tensor, action_idx, reward, next_state_tensor, done, next_legal_mask)

        self.steps_done += 1

        # Decay epsilon every step
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        # ── Skip optimisation during warmup ───────────────────────────────────
        if len(self.memory) < self.warmup_steps:
            return None

        loss_val = self._optimize_model()

        # ── Soft target update (Polyak) ────────────────────────────────────────
        if self.steps_done % self.target_update_interval == 0:
            self._soft_update_target()

        return loss_val

    def _soft_update_target(self):
        """θ_target ← τ·θ_policy + (1-τ)·θ_target"""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    # ── Optimisation ───────────────────────────────────────────────────────────

    def _optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size, self.device)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, next_legal_mask_batch = batch

        # Current Q values: Q_policy(s, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # ── Double DQN target ─────────────────────────────────────────────────
        # Step 1: use policy net to SELECT the best next action
        # Step 2: use target net to EVALUATE that action
        # → decouples selection from evaluation, reducing overestimation
        with torch.no_grad():
            next_q_policy = self.policy_net(next_state_batch)
            next_q_policy = next_q_policy.masked_fill(~next_legal_mask_batch, -float('inf'))
            no_legal_actions = ~next_legal_mask_batch.any(dim=1, keepdim=True)
            next_q_policy = next_q_policy.masked_fill(no_legal_actions, 0.0)

            next_actions = next_q_policy.argmax(dim=1, keepdim=True)
            next_state_values = self.target_net(next_state_batch).gather(1, next_actions)
            next_state_values = next_state_values.masked_fill(no_legal_actions, 0.0)

        expected_state_action_values = reward_batch + (
            self.gamma * next_state_values * (1.0 - done_batch)
        )

        # Huber loss (smooth_l1) is less sensitive to outliers than MSE
        loss = nn.functional.smooth_l1_loss(
            state_action_values, expected_state_action_values
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient norm clipping — more principled than per-param clamping
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()
