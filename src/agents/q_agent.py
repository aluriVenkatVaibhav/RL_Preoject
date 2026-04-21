import random

from game import Agent, Directions
from util import Counter, manhattanDistance
from utils.state_parser import StateParser


class QAgent(Agent):
    """
    Tabular Q-learning agent with a compact, task-focused state key.

    Key improvements over the original baseline:
      - uses config-driven epsilon start/end/decay
      - decays epsilon per episode in final()
      - avoids Stop when movement is available
      - uses a richer tabular state abstraction than the original flat tuple
    """

    def __init__(self, alpha=0.1, epsilon=0.05, gamma=0.99, numTraining=10, **kwargs):
        super(QAgent, self).__init__()
        self.alpha = float(kwargs.get('alpha', alpha))
        self.gamma = float(kwargs.get('gamma', gamma))
        self.numTraining = int(numTraining)

        self.epsilon = float(kwargs.get('epsilon_start', kwargs.get('epsilon', epsilon)))
        self.epsilon_min = float(kwargs.get('epsilon_end', self.epsilon))
        decay_episodes = max(1, int(kwargs.get('epsilon_decay_episodes', 1)))
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / decay_episodes

        self.avoid_stop = kwargs.get('avoid_stop', True)
        self.initial_q = float(kwargs.get('initial_q', 0.0))
        self.q_values = Counter()
        self.state_parser = StateParser()
        self.is_eval = False

    def getQValue(self, state_key, action):
        key = (state_key, action)
        if key not in self.q_values:
            return self.initial_q
        return self.q_values[key]

    def computeValueFromQValues(self, state, state_key):
        legal_actions = self._filtered_legal_actions(state)
        if not legal_actions:
            return 0.0
        return max(self.getQValue(state_key, action) for action in legal_actions)

    def computeActionFromQValues(self, state, state_key):
        legal_actions = self._filtered_legal_actions(state)
        if not legal_actions:
            return None

        best_value = self.computeValueFromQValues(state, state_key)
        best_actions = [
            action for action in legal_actions
            if self.getQValue(state_key, action) == best_value
        ]
        return random.choice(best_actions)

    def getAction(self, state):
        legal_actions = self._filtered_legal_actions(state)
        if not legal_actions:
            return None

        state_key = self.get_state_key(state)

        if not self.is_eval and random.random() < self.epsilon:
            return random.choice(legal_actions)
        return self.computeActionFromQValues(state, state_key)

    def update(self, state, action, nextState, reward, done=False, **kwargs):
        """
        Q-learning update:
            Q(s,a) <- Q(s,a) + alpha * (R + gamma * max_a' Q(s',a') - Q(s,a))
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(nextState)

        if done or not self._filtered_legal_actions(nextState):
            target = reward
        else:
            target = reward + self.gamma * self.computeValueFromQValues(nextState, next_state_key)

        current_q = self.getQValue(state_key, action)
        self.q_values[(state_key, action)] = current_q + self.alpha * (target - current_q)

    def final(self, state):
        if not self.is_eval:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def _filtered_legal_actions(self, state):
        legal_actions = state.getLegalActions(self.index)
        if self.avoid_stop and len(legal_actions) > 1 and Directions.STOP in legal_actions:
            return [action for action in legal_actions if action != Directions.STOP]
        return legal_actions

    def _nearest_target_features(self, origin, targets):
        if origin is None or not targets:
            return (0, 0, 0)

        target = min(targets, key=lambda pos: manhattanDistance(origin, pos))
        dx = int(target[0] - origin[0])
        dy = int(target[1] - origin[1])
        dist = manhattanDistance(origin, target)
        return (
            self._sign(dx),
            self._sign(dy),
            self._distance_bucket(dist),
        )

    def get_state_key(self, state):
        pac_pos = state.getPacmanPosition()
        if pac_pos is None:
            return tuple()

        food_list = state.getFood().asList()
        capsules = state.getCapsules()

        food_dx, food_dy, food_dist = self._nearest_target_features(pac_pos, food_list)
        cap_dx, cap_dy, cap_dist = self._nearest_target_features(pac_pos, capsules)

        ghost_data = []
        for ghost in state.getGhostStates():
            if ghost.configuration is None:
                continue
            ghost_pos = ghost.getPosition()
            ghost_data.append(
                (
                    manhattanDistance(pac_pos, ghost_pos),
                    int(ghost_pos[0] - pac_pos[0]),
                    int(ghost_pos[1] - pac_pos[1]),
                    int(ghost.scaredTimer > 0),
                )
            )

        if ghost_data:
            ghost_dist_raw, ghost_dx_raw, ghost_dy_raw, ghost_scared = min(
                ghost_data, key=lambda item: item[0]
            )
            ghost_dx = self._sign(ghost_dx_raw)
            ghost_dy = self._sign(ghost_dy_raw)
            ghost_dist = self._distance_bucket(ghost_dist_raw)
        else:
            ghost_dx, ghost_dy, ghost_dist, ghost_scared = 0, 0, 0, 0

        legal_actions = set(self._filtered_legal_actions(state))
        local_moves = (
            int(Directions.NORTH in legal_actions),
            int(Directions.SOUTH in legal_actions),
            int(Directions.EAST in legal_actions),
            int(Directions.WEST in legal_actions),
        )

        food_remaining_bucket = min(len(food_list) // 5, 10)
        capsule_count_bucket = min(len(capsules), 2)

        return (
            food_dx, food_dy, food_dist,
            cap_dx, cap_dy, cap_dist,
            ghost_dx, ghost_dy, ghost_dist, ghost_scared,
            food_remaining_bucket,
            capsule_count_bucket,
            *local_moves,
        )

    def _distance_bucket(self, distance):
        return min(int(distance // 2), 6)

    def _sign(self, value):
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.q_values, f)

    def load(self, path):
        import pickle
        import os
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.q_values = pickle.load(f)


class ApproxQAgent(QAgent):
    """
    Approximate Q-learning agent using feature-based weights.
    """
    def __init__(self, alpha=0.1, epsilon=0.05, gamma=0.99, numTraining=10, **kwargs):
        super(ApproxQAgent, self).__init__(alpha, epsilon, gamma, numTraining, **kwargs)
        self.weights = Counter()

    def getFeatures(self, state, action):
        """
        Extract simple features for state-action evaluation.
        Returns a dict of features {feature_name: value}.
        """
        features = Counter()
        features["bias"] = 1.0

        successor = state.generateSuccessor(self.index, action)
        features["score"] = successor.getScore()

        state_tuple = self.state_parser.get_flat_feature_vector(successor)
        if len(state_tuple) >= 6:
            features["food_dist"] = -state_tuple[2]
            features["ghost_dist"] = state_tuple[4]
            features["ghost_scared"] = state_tuple[5]

        return features

    def getQValue(self, state, action):
        features = self.getFeatures(state, action)
        q_value = 0.0
        for feature, val in features.items():
            q_value += self.weights[feature] * val
        return q_value

    def update(self, state, action, nextState, reward, done=False, **kwargs):
        features = self.getFeatures(state, action)

        if done or not self._filtered_legal_actions(nextState):
            target = reward
        else:
            max_q_next = max(
                self.getQValue(nextState, next_action)
                for next_action in self._filtered_legal_actions(nextState)
            )
            target = reward + self.gamma * max_q_next

        difference = target - self.getQValue(state, action)

        for feature, val in features.items():
            self.weights[feature] += self.alpha * difference * val

    def computeValueFromQValues(self, state, state_key=None):
        legal_actions = self._filtered_legal_actions(state)
        if not legal_actions:
            return 0.0
        return max(self.getQValue(state, action) for action in legal_actions)

    def computeActionFromQValues(self, state, state_key=None):
        legal_actions = self._filtered_legal_actions(state)
        if not legal_actions:
            return None
        best_value = self.computeValueFromQValues(state)
        best_actions = [a for a in legal_actions if self.getQValue(state, a) == best_value]
        return random.choice(best_actions)

    def getAction(self, state):
        legal_actions = self._filtered_legal_actions(state)
        if not legal_actions:
            return None

        if not self.is_eval and random.random() < self.epsilon:
            return random.choice(legal_actions)
        return self.computeActionFromQValues(state)

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)

    def load(self, path):
        import pickle
        import os
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.weights = pickle.load(f)
