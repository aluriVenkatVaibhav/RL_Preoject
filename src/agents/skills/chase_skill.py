from util import manhattanDistance
from .base_skill import BaseSkill


class ChaseSkill(BaseSkill):
    """
    Skill: Chase and eat nearby scared ghosts.
    Shaped reward: progress toward scared ghost + large bonus for eating one.
    """

    def get_features(self, state):
        pac_pos = state.getPacmanPosition()
        scared = [g for g in state.getGhostStates() if g.scaredTimer > 0 and g.configuration]
        if not scared:
            return (99, 0)
        nearest = min(scared, key=lambda g: manhattanDistance(pac_pos, g.getPosition()))
        dist = manhattanDistance(pac_pos, nearest.getPosition())
        dist_bucket = min(dist // 2, 8)
        return (dist_bucket, min(len(scared), 4))

    def get_shaped_reward(self, state, next_state, base_reward):
        pac_pos = state.getPacmanPosition()
        next_pac_pos = next_state.getPacmanPosition()
        scared = [g for g in state.getGhostStates() if g.scaredTimer > 0 and g.configuration]
        next_scared = [g for g in next_state.getGhostStates() if g.scaredTimer > 0 and g.configuration]

        reward = 0.0
        # Large bonus for eating a scared ghost
        if len(next_scared) < len(scared) and not next_state.isLose():
            reward += 200.0
        elif scared:
            prev_dist = min(manhattanDistance(pac_pos, g.getPosition()) for g in scared)
            if next_scared:
                next_dist = min(manhattanDistance(next_pac_pos, g.getPosition()) for g in next_scared)
                reward += (prev_dist - next_dist) * 2.0
        if next_state.isLose():
            reward -= 500.0
        reward -= 0.5
        return reward
