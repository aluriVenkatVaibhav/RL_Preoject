import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from game import Agent, Directions
from utils.state_parser import StateParser
from models.actor_critic_net import ActorCriticNet

class ReinforceAgent(Agent):
    """
    Monte Carlo Policy Gradient Agent (REINFORCE) with a value baseline.
    """
    def __init__(self, **kwargs):
        super(ReinforceAgent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = kwargs.get('gamma', 0.99)
        self.lr = kwargs.get('learning_rate', 5e-4)
        
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
        
        self.saved_log_probs = []
        self.rewards = []
        self.is_eval = False

    def init_network(self, state):
        if self.network is None:
            self.state_parser.update_dims(state)
            self.network = ActorCriticNet(self.state_parser.input_shape, self.num_actions).to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    def getAction(self, state):
        self.init_network(state)
        
        legal_actions = state.getLegalActions(self.index)
        if not legal_actions:
            return None
            
        legal_idx = [self.action_to_idx[a] for a in legal_actions]
        state_tensor = self.state_parser.get_tensor(state, self.device)
        
        logits, _ = self.network(state_tensor)
        logits = logits.squeeze(0)
        
        # Masking
        mask = torch.full((self.num_actions,), -float('inf')).to(self.device)
        for idx in legal_idx:
            mask[idx] = 0
            
        masked_logits = logits + mask
        probs = torch.softmax(masked_logits, dim=-1)
        
        m = Categorical(probs)
        
        if self.is_eval:
            action_idx = probs.argmax().item()
        else:
            action_idx = m.sample()
            self.saved_log_probs.append(m.log_prob(action_idx))
            
        return self.idx_to_action[action_idx.item()]

    def update(self, state, action, nextState, reward, done):
        """
        Record the reward to compute returns at the end of the episode.
        The actual update is performed in final().
        """
        if not self.is_eval:
            self.rewards.append(reward)

    def final(self, state):
        """
        Called at the end of the episode. We perform the update here.
        """
        if self.is_eval or len(self.saved_log_probs) == 0:
            return
            
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(self.device)
        # Normalize returns
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / returns.std()
            
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        loss = torch.stack(policy_loss).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        del self.saved_log_probs[:]
        del self.rewards[:]

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
