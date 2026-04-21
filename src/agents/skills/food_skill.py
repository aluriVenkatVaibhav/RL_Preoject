from util import manhattanDistance
from .base_skill import BaseSkill


class FoodSkill(BaseSkill):
    """
    Skill: Navigate to and eat the nearest food pellet.
    Shaped reward: progress toward nearest food + bonus for eating.
    """

    def get_features(self, state):
        pac_pos = state.getPacmanPosition()
        food_list = state.getFood().asList()
        if not food_list:
            return (0, 0)
        nearest = min(food_list, key=lambda f: manhattanDistance(pac_pos, f))
        dist = manhattanDistance(pac_pos, nearest)
        dist_bucket = min(dist // 3, 6)
        num_food_bucket = min(len(food_list) // 5, 5)
        return (dist_bucket, num_food_bucket)

    def get_shaped_reward(self, state, next_state, base_reward):
        pac_pos = state.getPacmanPosition()
        next_pac_pos = next_state.getPacmanPosition()
        food_list = state.getFood().asList()
        next_food_list = next_state.getFood().asList()

        reward = 0.0
        # Bonus for eating food
        if len(next_food_list) < len(food_list):
            reward += 10.0
        # Progress reward: getting closer to nearest food
        if food_list:
            prev_dist = min(manhattanDistance(pac_pos, f) for f in food_list)
            if next_food_list:
                next_dist = min(manhattanDistance(next_pac_pos, f) for f in next_food_list)
                reward += (prev_dist - next_dist) * 0.5
        # Death penalty
        if next_state.isLose():
            reward -= 500.0
        # Small step penalty to encourage efficiency
        reward -= 0.5
        return reward
