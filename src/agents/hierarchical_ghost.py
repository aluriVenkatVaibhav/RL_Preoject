import random
from collections import defaultdict

from util import manhattanDistance
from game import Agent, Directions


class HierarchicalGhostAgent(Agent):
    """
    Two-level hierarchical Ghost agent.
    High-level goals: chase_pacman | scatter | ambush
    Low-level: tabular Q-learning movement controller.
    """

    GOALS = ['chase_pacman', 'scatter', 'ambush']

    def __init__(self, index, **kwargs):
        super().__init__(index)
        self.goal_interval = kwargs.get('ghost_goal_interval', 15)
        self.epsilon = kwargs.get('ghost_epsilon', 0.2)
        self.alpha = kwargs.get('ghost_alpha', 0.2)
        self.gamma = kwargs.get('gamma', 0.95)

        self.q_table = defaultdict(float)
        self.current_goal = 'chase_pacman'
        self.goal_step_count = 0
        self.goal_counts = defaultdict(int)
        self.is_learning = False

    # ------------------------------------------------------------------
    # Goal selection (rule-based heuristic for the meta-level)
    # ------------------------------------------------------------------
    def _select_goal(self, state):
        if not state.getGhostState(self.index).configuration:
            return 'scatter'
        ghost_pos = state.getGhostPosition(self.index)
        pac_pos   = state.getPacmanPosition()
        dist = manhattanDistance(ghost_pos, pac_pos)
        if dist > 8:
            return 'ambush'
        return 'chase_pacman'

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _get_features(self, state, goal):
        pac_pos = state.getPacmanPosition()
        ghost_state = state.getGhostState(self.index)
        if not ghost_state.configuration:
            return (goal, 99)
        ghost_pos = state.getGhostPosition(self.index)
        dist = manhattanDistance(ghost_pos, pac_pos)
        dist_bucket = min(dist // 2, 8)
        return (goal, dist_bucket)

    # ------------------------------------------------------------------
    # Action selection (Q-learning epsilon-greedy)
    # ------------------------------------------------------------------
    def getAction(self, state):
        legal = [a for a in state.getLegalActions(self.index) if a != Directions.STOP]
        if not legal:
            return Directions.STOP

        # Refresh goal periodically
        if self.goal_step_count % self.goal_interval == 0:
            self.current_goal = self._select_goal(state)
            self.goal_counts[self.current_goal] += 1
        self.goal_step_count += 1

        if self.is_learning and random.random() < self.epsilon:
            return random.choice(legal)

        features = self._get_features(state, self.current_goal)
        return max(legal, key=lambda a: self.q_table[(features, a)])

    # ------------------------------------------------------------------
    # Q-learning update
    # ------------------------------------------------------------------
    def _shaped_reward(self, state, next_state, base_reward):
        """Distance-based shaping: reward getting closer to Pacman."""
        ghost_state = state.getGhostState(self.index)
        next_ghost_state = next_state.getGhostState(self.index)
        pac_pos = state.getPacmanPosition()
        reward = base_reward
        if ghost_state.configuration and next_ghost_state.configuration:
            prev_dist = manhattanDistance(ghost_state.getPosition(), pac_pos)
            next_dist = manhattanDistance(next_ghost_state.getPosition(), next_state.getPacmanPosition())
            reward += (prev_dist - next_dist) * 5.0   # positive if closing in
        if next_state.isLose():
            reward += 200.0   # caught Pacman
        reward -= 0.1         # step penalty
        return reward

    def update(self, state, action, next_state, reward, done=False, **kwargs):
        if not self.is_learning:
            return None
        features = self._get_features(state, self.current_goal)
        next_legal = [a for a in next_state.getLegalActions(self.index) if a != Directions.STOP]
        if next_legal and not done:
            next_features = self._get_features(next_state, self.current_goal)
            max_next_q = max(self.q_table[(next_features, a)] for a in next_legal)
        else:
            max_next_q = 0.0
        shaped = self._shaped_reward(state, next_state, reward)
        current_q = self.q_table[(features, action)]
        self.q_table[(features, action)] = current_q + self.alpha * (
            shaped + self.gamma * max_next_q - current_q
        )
        return None

    def final(self, state):
        pass

    def set_learning(self, is_learning):
        self.is_learning = is_learning

    def get_goal_stats(self):
        return dict(self.goal_counts)

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path):
        import pickle, os
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.q_table = defaultdict(float, pickle.load(f))
