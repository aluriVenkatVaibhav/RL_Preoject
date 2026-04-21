"""
hierarchical_train.py
=====================
4-Phase staged training pipeline for the Hierarchical Pacman/Ghost system.

  Phase 1 – Train Pacman SKILLS independently (ghosts frozen/baseline)
  Phase 2 – Train META-CONTROLLER with frozen skills
  Phase 3 – Train GHOST agents (Pacman frozen)
  Phase 4 – ALTERNATING: freeze ghosts/train Pacman, then freeze Pacman/train ghosts

Usage (from repo root):
  conda run -n Env python src/hierarchical_train.py --config src/config.yaml
  conda run -n Env python src/hierarchical_train.py --config src/config.yaml --ghost_mode hierarchical
"""
import argparse
import os
import sys

# Resolve --config before chdir so relative CLI paths work
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--config', type=str, default='config.yaml')
_pre_args, _ = _pre.parse_known_args()
_ABS_CONFIG = os.path.abspath(_pre_args.config)

# Always execute from src/
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SRC_DIR)
sys.path.insert(0, SRC_DIR)

import yaml
import matplotlib.pyplot as plt

from pacman import ClassicGameRules
import layout
from textDisplay import NullGraphics
from utils.logger import Logger
from utils.reward_shaper import shape_pacman_reward, shape_ghost_reward
from agents.hierarchical_pacman import HierarchicalPacmanAgent
from agents.hierarchical_ghost import HierarchicalGhostAgent
from ghostAgents import DirectionalGhost


# ── helpers ──────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_pacman(config):
    kw = {**config['hyperparameters'], **config.get('hierarchical', {})}
    return HierarchicalPacmanAgent(**kw)


def build_ghosts(num, mode, config):
    ghosts = []
    for i in range(num):
        idx = i + 1
        if mode == 'hierarchical':
            ghosts.append(HierarchicalGhostAgent(idx, **config['hyperparameters']))
        else:
            ghosts.append(DirectionalGhost(idx))
    return ghosts


# ── custom episode loop ───────────────────────────────────────────────────────

def run_hierarchical_episode(game, pacman, ghosts, rules):
    """
    Episode loop that uses shaped rewards and calls .update() / .final()
    on both Pacman and Ghost agents.
    """
    game.display.initialize(game.state.data)

    for agent in game.agents:
        if hasattr(agent, 'registerInitialState'):
            agent.registerInitialState(game.state.deepCopy())

    num_agents = len(game.agents)
    agent_idx  = game.startingIndex
    prev_states  = {i: None for i in range(num_agents)}
    prev_actions = {i: None for i in range(num_agents)}

    while not game.gameOver:
        agent = game.agents[agent_idx]
        state = game.state.deepCopy()
        done  = state.isWin() or state.isLose()

        if prev_states[agent_idx] is not None and hasattr(agent, 'update'):
            if agent_idx == 0:
                reward = shape_pacman_reward(prev_states[agent_idx], state)
            else:
                reward = shape_ghost_reward(prev_states[agent_idx], state, agent_idx)
            agent.update(prev_states[agent_idx], prev_actions[agent_idx], state, reward, done)

        if done:
            break

        action = agent.getAction(state)
        prev_states[agent_idx]  = state
        prev_actions[agent_idx] = action

        game.state = game.state.generateSuccessor(agent_idx, action)
        game.display.update(game.state.data)
        rules.process(game.state, game)
        agent_idx = (agent_idx + 1) % num_agents

    # Terminal updates
    final_state = game.state.deepCopy()
    for i, agent in enumerate(game.agents):
        if prev_states[i] is not None and hasattr(agent, 'update'):
            if i == 0:
                r = shape_pacman_reward(prev_states[i], final_state)
            else:
                r = shape_ghost_reward(prev_states[i], final_state, i)
            agent.update(prev_states[i], prev_actions[i], final_state, r, True)
        if hasattr(agent, 'final'):
            agent.final(final_state)

    game.display.finish()
    return game.state.getScore(), game.state.isWin()


# ── phase runner ─────────────────────────────────────────────────────────────

def run_phase(name, episodes, pacman, ghosts, lay, rules, logger,
              freeze_pacman=False, freeze_ghosts=True, alt_interval=None):
    print(f"\n{'='*55}")
    print(f"  {name}  ({episodes} eps)")
    print(f"{'='*55}")

    if alt_interval is None:   # fixed-freeze phase
        _apply_freeze(pacman, ghosts, freeze_pacman, freeze_ghosts)
        for ep in range(episodes):
            _run_one(ep, episodes, pacman, ghosts, lay, rules, logger)
    else:                      # alternating phase
        for ep in range(episodes):
            if (ep // alt_interval) % 2 == 0:
                _apply_freeze(pacman, ghosts, freeze_pacman=False, freeze_ghosts=True)
            else:
                _apply_freeze(pacman, ghosts, freeze_pacman=True, freeze_ghosts=False)
            _run_one(ep, episodes, pacman, ghosts, lay, rules, logger)


def _apply_freeze(pacman, ghosts, freeze_pacman, freeze_ghosts):
    pacman.is_eval = freeze_pacman
    for sk in pacman.skills.values():
        sk.is_eval = freeze_pacman
    for g in ghosts:
        if hasattr(g, 'set_learning'):
            g.set_learning(not freeze_ghosts)


def _run_one(ep, total, pacman, ghosts, lay, rules, logger):
    game = rules.newGame(lay, pacman, ghosts, NullGraphics(), quiet=True, catchExceptions=False)
    score, is_win = run_hierarchical_episode(game, pacman, ghosts, rules)
    length = len(game.moveHistory)
    logger.log_episode(score, length, is_win)

    if (ep + 1) % 10 == 0:
        avg = sum(logger.metrics['episode_rewards'][-10:]) / 10
        wr  = logger.metrics['win_rates'][-1]
        gs  = pacman.get_goal_stats() if not pacman.is_eval else {}
        print(f"  [{ep+1:>5}/{total}] AvgR: {avg:8.1f} | WinRate: {wr:.2f} | Goals: {gs}")


# ── plot helpers ─────────────────────────────────────────────────────────────

def plot_goal_freq(goal_stats, plots_dir, prefix='hierarchical'):
    if not goal_stats:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(goal_stats.keys())
    values = list(goal_stats.values())
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    ax.bar(labels, values, color=colors[:len(labels)])
    ax.set_title('Goal Selection Frequency', fontsize=14)
    ax.set_xlabel('Goal')
    ax.set_ylabel('Total selections')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{prefix}_goal_freq.png'))
    plt.close()


def plot_skill_usage(skill_stats, plots_dir, prefix='hierarchical'):
    if not skill_stats:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(skill_stats.keys())
    values = list(skill_stats.values())
    ax.bar(labels, values, color='steelblue')
    ax.set_title('Skill Step Usage', fontsize=14)
    ax.set_xlabel('Skill')
    ax.set_ylabel('Steps executed')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{prefix}_skill_usage.png'))
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--ghost_mode', type=str, default='baseline',
                        choices=['baseline', 'hierarchical'])
    args = parser.parse_args()

    # Use the pre-resolved absolute path (resolved before chdir)
    config = load_config(_ABS_CONFIG)

    results_dir = config['training']['results_dir']
    plots_dir   = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    logger = Logger(results_dir)
    lay    = layout.getLayout(config['env']['layout'])
    if lay is None:
        raise RuntimeError(f"Layout '{config['env']['layout']}' not found.")
    rules  = ClassicGameRules(timeout=config['env']['timeout'])
    rules.quiet = True

    pacman = build_pacman(config)
    ghosts = build_ghosts(config['env']['num_ghosts'], args.ghost_mode, config)

    total_eps   = config['training']['num_episodes']
    per_phase   = max(1, total_eps // 4)
    alt_interval = config['multiagent'].get('alternating_interval', 50)

    # ── Phase 1: train skills ────────────────────────────────────────────────
    run_phase("Phase 1 – Skill Training", per_phase,
              pacman, ghosts, lay, rules, logger,
              freeze_pacman=False, freeze_ghosts=True)
    for goal, skill in pacman.skills.items():
        skill.save(os.path.join(results_dir, f'skill_{goal}.pkl'))

    # ── Phase 2: freeze skills, train meta-controller ────────────────────────
    for sk in pacman.skills.values():
        sk.is_eval = True
    run_phase("Phase 2 – Meta-Controller Training", per_phase,
              pacman, ghosts, lay, rules, logger,
              freeze_pacman=False, freeze_ghosts=True)
    for sk in pacman.skills.values():
        sk.is_eval = False   # unfreeze for remaining phases

    # ── Phase 3: freeze Pacman, train ghosts ─────────────────────────────────
    run_phase("Phase 3 – Ghost Training", per_phase,
              pacman, ghosts, lay, rules, logger,
              freeze_pacman=True, freeze_ghosts=False)

    # ── Phase 4: alternating joint training ──────────────────────────────────
    run_phase("Phase 4 – Alternating Joint Training", per_phase,
              pacman, ghosts, lay, rules, logger,
              alt_interval=alt_interval)

    # ── Save everything ──────────────────────────────────────────────────────
    logger.save_logs(prefix='hierarchical')
    logger.plot_metrics(prefix='hierarchical')
    pacman.save(os.path.join(results_dir, 'hierarchical_pacman.pt'))
    for i, ghost in enumerate(ghosts):
        if hasattr(ghost, 'save'):
            ghost.save(os.path.join(results_dir, f'hierarchical_ghost_{i+1}.pkl'))

    # ── Extra plots ──────────────────────────────────────────────────────────
    plot_goal_freq(pacman.get_goal_stats(),       plots_dir)
    plot_skill_usage(pacman.get_skill_step_stats(), plots_dir)

    print("\n✓ Hierarchical Training Complete! Results saved to:", results_dir)


if __name__ == '__main__':
    main()
