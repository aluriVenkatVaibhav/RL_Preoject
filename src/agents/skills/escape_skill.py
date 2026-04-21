from util import manhattanDistance
from .base_skill import BaseSkill


class EscapeSkill(BaseSkill):
    """
    Skill: Evade nearby dangerous (non-scared) ghosts.
    Shaped reward: reward for increasing distance from nearest ghost.
    """

    def get_features(self, state):
        pac_pos = state.getPacmanPosition()
        dangerous = [g for g in state.getGhostStates() if g.scaredTimer == 0 and g.configuration]
        if not dangerous:
            return (99, 0)
        nearest_dist = min(manhattanDistance(pac_pos, g.getPosition()) for g in dangerous)
        dist_bucket = min(nearest_dist // 2, 8)
        return (dist_bucket, min(len(dangerous), 4))

    def get_shaped_reward(self, state, next_state, base_reward):
        pac_pos = state.getPacmanPosition()
        next_pac_pos = next_state.getPacmanPosition()
        dangerous = [g for g in state.getGhostStates() if g.scaredTimer == 0 and g.configuration]

        reward = 0.0
        if dangerous:
            prev_dist = min(manhattanDistance(pac_pos, g.getPosition()) for g in dangerous)
            next_dangerous = [g for g in next_state.getGhostStates() if g.scaredTimer == 0 and g.configuration]
            if next_dangerous:
                next_dist = min(manhattanDistance(next_pac_pos, g.getPosition()) for g in next_dangerous)
                # Reward for moving AWAY from ghost
                reward += (next_dist - prev_dist) * 2.0
        # Heavy penalty for dying
        if next_state.isLose():
            reward -= 500.0
        reward -= 0.3
        return reward
