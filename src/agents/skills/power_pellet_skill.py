from util import manhattanDistance
from .base_skill import BaseSkill


class PowerPelletSkill(BaseSkill):
    """
    Skill: Navigate to and eat the nearest power pellet (capsule).
    Shaped reward: progress toward pellet + large bonus for eating.
    """

    def get_features(self, state):
        pac_pos = state.getPacmanPosition()
        capsules = state.getCapsules()
        if not capsules:
            return (99, 0)
        nearest = min(capsules, key=lambda c: manhattanDistance(pac_pos, c))
        dist = manhattanDistance(pac_pos, nearest)
        dist_bucket = min(dist // 2, 8)
        return (dist_bucket, len(capsules))

    def get_shaped_reward(self, state, next_state, base_reward):
        pac_pos = state.getPacmanPosition()
        next_pac_pos = next_state.getPacmanPosition()
        capsules = state.getCapsules()
        next_capsules = next_state.getCapsules()

        reward = 0.0
        # Large bonus for eating a power pellet
        if len(next_capsules) < len(capsules):
            reward += 50.0
        elif capsules:
            prev_dist = min(manhattanDistance(pac_pos, c) for c in capsules)
            remaining = [c for c in capsules if c in next_capsules]
            if remaining:
                next_dist = min(manhattanDistance(next_pac_pos, c) for c in remaining)
                reward += (prev_dist - next_dist) * 1.0
        if next_state.isLose():
            reward -= 500.0
        reward -= 0.5
        return reward
