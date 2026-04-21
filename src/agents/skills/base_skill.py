import random
from abc import ABC, abstractmethod
from collections import defaultdict
from game import Directions


class BaseSkill(ABC):
    """
    Abstract base class for all low-level skill agents.
    Skills use lightweight tabular Q-learning over hand-crafted features.
    """

    def __init__(self, skill_alpha=0.3, skill_epsilon=0.15, skill_gamma=0.95, **kwargs):
        self.alpha = skill_alpha
        self.epsilon = skill_epsilon
        self.gamma = skill_gamma
        self.weights = defaultdict(float)
        self.is_eval = False
        self.step_count = 0

    @abstractmethod
    def get_features(self, state):
        """Return a hashable feature tuple describing the state for this skill."""
        pass

    @abstractmethod
    def get_shaped_reward(self, state, next_state, base_reward):
        """Return intrinsic shaped reward for this skill's specific goal."""
        pass

    def get_q_value(self, features, action):
        return self.weights[(features, action)]

    def get_best_action(self, state, legal_actions):
        features = self.get_features(state)
        shuffled = list(legal_actions)
        random.shuffle(shuffled)   # break ties randomly
        return max(shuffled, key=lambda a: self.get_q_value(features, a))

    def get_action(self, state, legal_actions):
        """Epsilon-greedy action selection."""
        if not legal_actions:
            return Directions.STOP
        if not self.is_eval and random.random() < self.epsilon:
            return random.choice(legal_actions)
        return self.get_best_action(state, legal_actions)

    def update(self, state, action, next_state, reward):
        """Tabular Q-learning update."""
        if self.is_eval:
            return
        features = self.get_features(state)
        next_legal = next_state.getLegalActions(0)
        if next_legal:
            next_features = self.get_features(next_state)
            max_next_q = max(self.get_q_value(next_features, a) for a in next_legal)
        else:
            max_next_q = 0.0
        current_q = self.get_q_value(features, action)
        self.weights[(features, action)] += self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.step_count += 1

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(self.weights), f)

    def load(self, path):
        import pickle, os
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.weights = defaultdict(float, pickle.load(f))
