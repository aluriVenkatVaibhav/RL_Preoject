"""
Shaped reward functions for both Pacman and Ghost hierarchical agents.

Changes vs baseline:
  - Added win_bonus (+500)  : explicit terminal reward for clearing the board
  - Added proximity_food (+2): potential-based shaping toward nearest food pellet
  - step_penalty kept at -1 (still needed to discourage loitering)
  - death kept at -500      : dying should be strongly penalised
"""
from util import manhattanDistance

# ── Pacman reward coefficients ──────────────────────────────────────────────
PACMAN = {
    'eat_food':        10.0,
    'eat_power_pellet': 50.0,
    'eat_ghost':       200.0,
    'win_bonus':       500.0,   # NEW: explicit win reward
    'death':          -500.0,
    'step_penalty':    -1.0,
    'proximity_food':   2.0,    # NEW: reward per Manhattan-cell closer to food
}

# ── Ghost reward coefficients ────────────────────────────────────────────────
GHOST = {
    'catch_pacman':    200.0,
    'proximity_bonus':   5.0,   # per step closer to Pacman
    'step_penalty':     -0.1,
}


def shape_pacman_reward(state, next_state):
    """
    Compute the shaped reward for the Pacman agent between two game states.

    Signals provided (in priority order):
      1. Win / lose terminal bonuses (large magnitude → clear Q-value separation)
      2. Event rewards: food eaten, capsule eaten, ghost eaten
      3. Potential-based food proximity (dense gradient toward nearest food)
      4. Step penalty (discourages loitering)
    """
    reward = 0.0

    # ── Terminal conditions (computed first — dominate all step rewards) ─────

    # Win: Pacman cleared the board
    if next_state.isWin():
        reward += PACMAN['win_bonus']
        reward += PACMAN['step_penalty']
        return reward   # no need to compute other signals for terminal state

    # Loss: Pacman was caught
    if next_state.isLose():
        reward += PACMAN['death']
        reward += PACMAN['step_penalty']
        return reward

    # ── Non-terminal step rewards ────────────────────────────────────────────

    # Food eaten
    food_diff = len(state.getFood().asList()) - len(next_state.getFood().asList())
    reward += food_diff * PACMAN['eat_food']

    # Power pellet eaten
    cap_diff = len(state.getCapsules()) - len(next_state.getCapsules())
    reward += cap_diff * PACMAN['eat_power_pellet']

    # Scared ghost eaten (scared count drops without Pacman dying)
    prev_scared = sum(1 for g in state.getGhostStates() if g.scaredTimer > 0)
    next_scared = sum(1 for g in next_state.getGhostStates() if g.scaredTimer > 0)
    if next_scared < prev_scared:
        reward += PACMAN['eat_ghost']

    # ── Potential-based food proximity (dense exploration gradient) ──────────
    # Reward = γ·Φ(s') − Φ(s), where Φ(s) = −min_distance_to_food
    # Simplified: positive when Pacman moves CLOSER to the nearest food pellet.
    pac_prev = state.getPacmanPosition()
    pac_next = next_state.getPacmanPosition()
    food_list = state.getFood().asList()   # food present BEFORE this action

    if pac_prev and pac_next and food_list:
        dist_before = min(manhattanDistance(pac_prev, f) for f in food_list)
        dist_after  = min(manhattanDistance(pac_next, f) for f in food_list)
        reward += (dist_before - dist_after) * PACMAN['proximity_food']

    # Step penalty
    reward += PACMAN['step_penalty']

    return reward


def shape_ghost_reward(state, next_state, ghost_index):
    """
    Compute the shaped reward for a specific ghost between two game states.
    """
    reward = GHOST['step_penalty']

    ghost_state      = state.getGhostState(ghost_index)
    next_ghost_state = next_state.getGhostState(ghost_index)

    if ghost_state.configuration and next_ghost_state.configuration:
        prev_dist = manhattanDistance(ghost_state.getPosition(), state.getPacmanPosition())
        next_dist = manhattanDistance(next_ghost_state.getPosition(), next_state.getPacmanPosition())
        reward += (prev_dist - next_dist) * GHOST['proximity_bonus']

    # Caught Pacman
    if next_state.isLose():
        reward += GHOST['catch_pacman']

    return reward
