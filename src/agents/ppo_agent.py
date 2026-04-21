import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from game import Agent, Directions
from utils.state_parser import StateParser
from models.actor_critic_net import ActorCriticNet

class PpoAgent(Agent):
    """
    Proximal Policy Optimization (PPO) Agent.
    """
    def __init__(self, **kwargs):
        super(PpoAgent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = kwargs.get('gamma', 0.99)
        self.lr = kwargs.get('ppo_learning_rate', kwargs.get('learning_rate', 3e-4))
        self.lr_end = kwargs.get('ppo_learning_rate_end', self.lr)
        self.lr_decay_updates = max(1, kwargs.get('ppo_lr_decay_updates', 300))
        
        self.ppo_epochs = kwargs.get('ppo_epochs', 4)
        self.clip_param = kwargs.get('ppo_clip', 0.2)
        self.value_clip = kwargs.get('ppo_value_clip', 0.2)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.entropy_coef_start = kwargs.get('ppo_entropy_coef_start', kwargs.get('entropy_coef', 0.01))
        self.entropy_coef_end = kwargs.get('ppo_entropy_coef_end', self.entropy_coef_start)
        self.entropy_decay_updates = max(1, kwargs.get('ppo_entropy_decay_updates', 300))
        self.entropy_coef = self.entropy_coef_start
        self.value_loss_coef = kwargs.get('value_loss_coef', 0.5)
        self.batch_size = kwargs.get('ppo_batch_size', kwargs.get('batch_size', 64))
        self.update_episodes = kwargs.get('ppo_update_episodes', 5)
        self.reward_scale = kwargs.get('ppo_reward_scale', kwargs.get('reward_scale', 0.1))
        self.reward_clip = kwargs.get('ppo_reward_clip', kwargs.get('reward_clip', 50.0))
        self.avoid_stop = kwargs.get('ppo_avoid_stop', kwargs.get('avoid_stop', True))
        self.max_grad_norm = kwargs.get('ppo_max_grad_norm', 0.5)
        
        self.state_parser = StateParser()
        
        self.action_to_idx = {
            Directions.NORTH: 0,
            Directions.SOUTH: 1,
            Directions.EAST: 2,
            Directions.WEST: 3,
            Directions.STOP: 4
        }
        self.idx_to_action = {v: k for k, v in self.action_to_idx.items()}
        self.num_actions = len(self.action_to_idx)
        
        self.network = None
        self.optimizer = None
        self.is_eval = False
        
        self.trajectory = []
        self.last_state = None
        self.last_action_idx = None
        self.last_log_prob = None
        self.last_value = None
        self.last_legal_mask = None
        self.last_loss = None
        self.last_entropy = None
        self.rollout = []
        self.episodes_since_update = 0
        self.update_steps = 0


    def getAction(self, state):
        self.init_network(state)
        
        legal_actions = self._filtered_legal_actions(state)
        if not legal_actions:
            return None
            
        legal_idx = [self.action_to_idx[a] for a in legal_actions]
        state_tensor = self.state_parser.get_tensor(state, self.device)
        
        with torch.no_grad():
            logits, value = self.network(state_tensor)
            
        logits = logits.squeeze(0)
        value = value.squeeze(0)
        
        mask = torch.full((self.num_actions,), -float('inf')).to(self.device)
        for idx in legal_idx:
            mask[idx] = 0
            
        masked_logits = logits + mask
        dist = Categorical(logits=masked_logits)
        
        if self.is_eval:
            action_idx = masked_logits.argmax().item()
            return self.idx_to_action[action_idx]
            
        action_idx = dist.sample()
        
        self.last_state = state_tensor.squeeze(0).cpu() # save as unbatched
        self.last_action_idx = action_idx.item()
        self.last_log_prob = dist.log_prob(action_idx).item()
        self.last_value = value.item()
        
        # Save legal actions mask to prevent selecting illegal actions during update
        self.last_legal_mask = mask.cpu()
        
        return self.idx_to_action[self.last_action_idx]

    def _filtered_legal_actions(self, state):
        legal_actions = state.getLegalActions(self.index)
        if self.avoid_stop and len(legal_actions) > 1 and Directions.STOP in legal_actions:
            return [action for action in legal_actions if action != Directions.STOP]
        return legal_actions

    def _scale_reward(self, reward):
        reward = float(reward) * self.reward_scale
        if self.reward_clip is not None:
            reward = max(-self.reward_clip, min(self.reward_clip, reward))
        return reward

    def update(self, state, action, nextState, reward, done):
        if self.is_eval or self.last_state is None:
            return None
        reward = self._scale_reward(reward)
        
        self.trajectory.append((
            self.last_state,
            self.last_action_idx,
            self.last_log_prob,
            self.last_value,
            reward,
            done,
            self.last_legal_mask
        ))

    def final(self, state):
        """
        Called at episode end. PPO trains locally per episode in this framework for simplicity.
        """
        if self.is_eval or not self.trajectory:
            return
            
        # Get value for final state
        final_done = state.isWin() or state.isLose()
        if final_done:
            next_value = 0.0
        else:
            state_tensor = self.state_parser.get_tensor(state, self.device)
            with torch.no_grad():
                _, next_value = self.network(state_tensor)
            next_value = next_value.item()
        
        states, actions, old_log_probs, values, rewards, dones, masks = zip(*self.trajectory)
        
        values = list(values) + [next_value]
        
        # Compute GAE
        returns = []
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            
        for sample in zip(states, actions, old_log_probs, values[:-1], returns, advantages, masks):
            self.rollout.append(sample)

        self.trajectory = []
        self.last_state = None
        self.last_action_idx = None
        self.last_log_prob = None
        self.last_value = None
        self.last_legal_mask = None
        self.episodes_since_update += 1

        if self.episodes_since_update < self.update_episodes:
            return None

        return self._update_from_rollout()

    def _update_from_rollout(self):
        if not self.rollout:
            return None

        self._update_entropy_coef()

        self._update_learning_rate()

        states, actions, old_log_probs, old_values, returns, advantages, masks = zip(*self.rollout)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        old_values = torch.tensor(old_values, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        masks = torch.stack(masks).to(self.device)

        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        total_entropy = 0.0
        update_count = 0

        for _ in range(self.ppo_epochs):
            sampler = BatchSampler(SubsetRandomSampler(range(len(states))), self.batch_size, drop_last=False)
            for indices in sampler:
                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_old_log_probs = old_log_probs[indices]
                batch_old_values = old_values[indices]
                batch_returns = returns[indices]
                batch_advantages = advantages[indices]
                batch_masks = masks[indices]
                
                logits, values_pred = self.network(batch_states)
                values_pred = values_pred.squeeze(-1)
                
                masked_logits = logits + batch_masks
                dist = Categorical(logits=masked_logits)
                
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                value_pred_clipped = batch_old_values + torch.clamp(
                    values_pred - batch_old_values,
                    -self.value_clip,
                    self.value_clip,
                )
                critic_loss_unclipped = nn.functional.smooth_l1_loss(values_pred, batch_returns, reduction='none')
                critic_loss_clipped = nn.functional.smooth_l1_loss(value_pred_clipped, batch_returns, reduction='none')
                critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped).mean()
                
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_entropy += entropy.item()
                update_count += 1
                
        self.rollout = []
        self.episodes_since_update = 0
        self.last_loss = total_loss / max(1, update_count)
        self.last_entropy = total_entropy / max(1, update_count)
        self.update_steps += 1
        return self.last_loss, self.last_entropy

    def _update_entropy_coef(self):
        progress = min(1.0, self.update_steps / self.entropy_decay_updates)
        self.entropy_coef = (
            self.entropy_coef_start
            + progress * (self.entropy_coef_end - self.entropy_coef_start)
        )

    def _update_learning_rate(self):
        progress = min(1.0, self.update_steps / self.lr_decay_updates)
        current_lr = self.lr + progress * (self.lr_end - self.lr)
        for group in self.optimizer.param_groups:
            group['lr'] = current_lr

    def save(self, path):
        if self.network is not None:
            torch.save(self.network.state_dict(), path)
            
    def load(self, path):
        import os
        if os.path.exists(path):
            self.load_path = path

    def init_network(self, state):
        if self.network is None:
            self.state_parser.update_dims(state)
            self.network = ActorCriticNet(self.state_parser.input_shape, self.num_actions).to(self.device)
            if hasattr(self, 'load_path') and self.load_path:
                self.network.load_state_dict(torch.load(self.load_path, map_location=self.device))
                self.load_path = None
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
