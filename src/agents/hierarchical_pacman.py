import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

from game import Agent, Directions
from utils.state_parser import StateParser
from utils.replay_buffer import ReplayBuffer
from models.dqn_net import DQNNet
from agents.skills.food_skill import FoodSkill
from agents.skills.power_pellet_skill import PowerPelletSkill
from agents.skills.chase_skill import ChaseSkill
from agents.skills.escape_skill import EscapeSkill


class HierarchicalPacmanAgent(Agent):
    """
    Two-level hierarchical Pacman agent:
      - Meta-Controller (DQN): selects high-level goal every `goal_interval` steps.
      - Skill Agents (tabular Q-learning): execute primitive actions toward current goal.
    
    Goals: eat_food | eat_power_pellet | chase_ghost | escape_ghost
    """

    GOALS = ['eat_food', 'eat_power_pellet', 'chase_ghost', 'escape_ghost']
    NUM_GOALS = 4

    def __init__(self, **kwargs):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ---- Meta-controller hyper-params ----
        self.goal_interval = kwargs.get('goal_interval', 5)
        self.meta_lr = kwargs.get('meta_lr', 1e-4)
        self.meta_gamma = kwargs.get('meta_gamma', 0.99)
        self.meta_epsilon = kwargs.get('meta_epsilon_start', 1.0)
        self.meta_epsilon_min = kwargs.get('meta_epsilon_end', 0.1)
        self.meta_epsilon_decay = (
            (self.meta_epsilon - self.meta_epsilon_min)
            / kwargs.get('meta_epsilon_decay_steps', 5000)
        )
        self.meta_batch_size = kwargs.get('batch_size', 64)
        self.target_update_interval = kwargs.get('target_update_interval', 200)

        # ---- Skill agents ----
        skill_kw = dict(
            skill_alpha=kwargs.get('skill_alpha', 0.3),
            skill_epsilon=kwargs.get('skill_epsilon', 0.15),
            skill_gamma=kwargs.get('skill_gamma', 0.95),
        )
        self.skills = {
            'eat_food':          FoodSkill(**skill_kw),
            'eat_power_pellet':  PowerPelletSkill(**skill_kw),
            'chase_ghost':       ChaseSkill(**skill_kw),
            'escape_ghost':      EscapeSkill(**skill_kw),
        }

        # ---- State parser for meta-controller ----
        self.state_parser = StateParser()

        # ---- Networks (lazy init on first state) ----
        self.meta_net = None
        self.meta_target_net = None
        self.meta_optimizer = None
        self.meta_memory = ReplayBuffer(kwargs.get('buffer_capacity', 20000))
        self.meta_steps = 0

        # ---- Episode bookkeeping ----
        self.current_goal = None
        self.goal_step_count = 0
        self.last_meta_state = None   # CPU tensor, unbatched
        self.last_goal_idx = None
        self.meta_reward_accum = 0.0

        # ---- Statistics ----
        self.goal_counts = defaultdict(int)
        self.skill_step_counts = defaultdict(int)

        self.is_eval = False
        self.step = 0

    # ------------------------------------------------------------------
    # Network initialisation
    # ------------------------------------------------------------------
    def _init_meta(self, state):
        if self.meta_net is None:
            self.state_parser.update_dims(state)
            shape = self.state_parser.input_shape
            self.meta_net = DQNNet(shape, self.NUM_GOALS).to(self.device)
            self.meta_target_net = DQNNet(shape, self.NUM_GOALS).to(self.device)
            if hasattr(self, '_meta_load_path') and self._meta_load_path:
                self.meta_net.load_state_dict(
                    torch.load(self._meta_load_path, map_location=self.device))
                self._meta_load_path = None
            self.meta_target_net.load_state_dict(self.meta_net.state_dict())
            self.meta_target_net.eval()
            self.meta_optimizer = optim.Adam(self.meta_net.parameters(), lr=self.meta_lr)

    # ------------------------------------------------------------------
    # Goal selection
    # ------------------------------------------------------------------
    def _should_reselect_goal(self, state):
        if self.current_goal is None:
            return True
        if self.goal_step_count >= self.goal_interval:
            return True
        # Goal no longer achievable
        if self.current_goal == 'eat_power_pellet' and not state.getCapsules():
            return True
        if self.current_goal == 'chase_ghost':
            if not any(g.scaredTimer > 0 for g in state.getGhostStates()):
                return True
        return False

    def _select_goal(self, state):
        """Epsilon-greedy goal selection via meta DQN."""
        state_t = self.state_parser.get_tensor(state, self.device)
        if not self.is_eval and random.random() < self.meta_epsilon:
            return random.randint(0, self.NUM_GOALS - 1), state_t
        with torch.no_grad():
            q = self.meta_net(state_t).squeeze(0)
        return int(q.argmax().item()), state_t

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def getAction(self, state):
        self._init_meta(state)

        legal = state.getLegalActions(self.index)
        if not legal:
            return Directions.STOP

        # ----- Meta-controller -----
        if self._should_reselect_goal(state):
            goal_idx, meta_st = self._select_goal(state)

            # Store previous meta-transition
            if self.last_meta_state is not None and not self.is_eval:
                done = state.isWin() or state.isLose()
                self.meta_memory.push(
                    self.last_meta_state,
                    self.last_goal_idx,
                    self.meta_reward_accum,
                    self.state_parser.get_tensor(state, 'cpu').squeeze(0),
                    done,
                )
                self._update_meta()

            self.current_goal = self.GOALS[goal_idx]
            self.goal_step_count = 0
            self.meta_reward_accum = 0.0
            self.last_meta_state = self.state_parser.get_tensor(state, 'cpu').squeeze(0)
            self.last_goal_idx = goal_idx
            self.goal_counts[self.current_goal] += 1

        self.goal_step_count += 1
        self.skill_step_counts[self.current_goal] += 1

        # ----- Execute active skill -----
        action = self.skills[self.current_goal].get_action(state, legal)
        self.step += 1
        return action

    # ------------------------------------------------------------------
    # Meta-controller DQN update
    # ------------------------------------------------------------------
    def _update_meta(self):
        if len(self.meta_memory) < self.meta_batch_size:
            return
        s, a, r, ns, d = self.meta_memory.sample(self.meta_batch_size, self.device)
        q_vals = self.meta_net(s).gather(1, a)
        with torch.no_grad():
            next_q = self.meta_target_net(ns).max(1)[0].unsqueeze(1)
        target = r + self.meta_gamma * next_q * (1 - d)
        loss = nn.functional.smooth_l1_loss(q_vals, target)
        self.meta_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.meta_net.parameters(), 1.0)
        self.meta_optimizer.step()
        self.meta_steps += 1
        self.meta_epsilon = max(self.meta_epsilon_min, self.meta_epsilon - self.meta_epsilon_decay)
        if self.meta_steps % self.target_update_interval == 0:
            self.meta_target_net.load_state_dict(self.meta_net.state_dict())

    # ------------------------------------------------------------------
    # Transition update (called every step)
    # ------------------------------------------------------------------
    def update(self, state, action, next_state, reward, done=False, **kwargs):
        """Update active skill Q-table and accumulate reward for meta-controller."""
        if self.is_eval or self.current_goal is None:
            return None
        skill = self.skills[self.current_goal]
        shaped = skill.get_shaped_reward(state, next_state, reward)
        skill.update(state, action, next_state, shaped)
        self.meta_reward_accum += reward   # meta uses original env reward
        return None

    # ------------------------------------------------------------------
    # Episode end
    # ------------------------------------------------------------------
    def final(self, state):
        if self.is_eval:
            return
        if self.last_meta_state is not None:
            final_t = self.state_parser.get_tensor(state, 'cpu').squeeze(0)
            self.meta_memory.push(
                self.last_meta_state,
                self.last_goal_idx,
                self.meta_reward_accum,
                final_t,
                True,
            )
            self._update_meta()
        # Reset episode state
        self.current_goal = None
        self.goal_step_count = 0
        self.last_meta_state = None
        self.last_goal_idx = None
        self.meta_reward_accum = 0.0

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_goal_stats(self):
        return dict(self.goal_counts)

    def get_skill_step_stats(self):
        return dict(self.skill_step_counts)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path):
        base = path.replace('.pt', '')
        if self.meta_net is not None:
            torch.save(self.meta_net.state_dict(), f"{base}_meta.pt")
        for goal, skill in self.skills.items():
            skill.save(f"{base}_skill_{goal}.pkl")

    def load(self, path):
        base = path.replace('.pt', '')
        meta_path = f"{base}_meta.pt"
        import os
        if os.path.exists(meta_path):
            self._meta_load_path = meta_path
        for goal, skill in self.skills.items():
            skill.load(f"{base}_skill_{goal}.pkl")
