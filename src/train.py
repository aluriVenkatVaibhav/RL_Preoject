import argparse
import yaml
import os
import random
import torch
from datetime import datetime
from time import perf_counter

from pacman import ClassicGameRules, GameState
import layout
from textDisplay import NullGraphics
from utils.logger import Logger
from utils.reward_shaper import shape_pacman_reward

# Import Agents
from agents.q_agent import QAgent, ApproxQAgent
from agents.sarsa_agent import SARSAAgent
from agents.dqn_agent import DQNAgent
from agents.reinforce_agent import ReinforceAgent
from agents.ppo_agent import PpoAgent
from agents.ghost_agent import RLGhostWrapper

from ghostAgents import RandomGhost, DirectionalGhost

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_pacman_agent(agent_type, config):
    kwargs = config['hyperparameters']
    if agent_type == "q_learning":
        return QAgent(**kwargs)
    elif agent_type == "sarsa":
        return SARSAAgent(**kwargs)
    elif agent_type == "approx_q":
        return ApproxQAgent(**kwargs)
    elif agent_type == "dqn":
        return DQNAgent(**kwargs)
    elif agent_type == "reinforce":
        return ReinforceAgent(**kwargs)
    elif agent_type == "ppo":
        return PpoAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type {agent_type}")

def create_ghost_agents(num_ghosts, mode, config):
    ghosts = []
    for i in range(num_ghosts):
        index = i + 1
        if mode == "baseline":
            # Rule based ghosts
            ghosts.append(DirectionalGhost(index))
        elif mode == "rl_q":
            ghosts.append(RLGhostWrapper(index, learner_class="QAgent", **config['hyperparameters']))
        else:
            ghosts.append(RandomGhost(index))
    return ghosts

def get_exploration_value(agent):
    for attr in ('epsilon', 'meta_epsilon', 'skill_epsilon', 'ghost_epsilon'):
        if hasattr(agent, attr):
            value = getattr(agent, attr)
            if isinstance(value, (int, float)):
                return value
    return None

def get_entropy_value(agent, episode_entropy):
    if episode_entropy is not None:
        return episode_entropy
    value = getattr(agent, 'last_entropy', None)
    if isinstance(value, (int, float)):
        return value
    return None

def get_entropy_coef(agent):
    value = getattr(agent, 'entropy_coef', None)
    if isinstance(value, (int, float)):
        return value
    return None

def get_learning_rate(agent):
    optimizer = getattr(agent, 'optimizer', None)
    if optimizer is None or not optimizer.param_groups:
        return None
    value = optimizer.param_groups[0].get('lr')
    if isinstance(value, (int, float)):
        return value
    return None

def uses_shaped_pacman_reward(agent_type):
    return agent_type in {'dqn', 'ppo', 'q_learning', 'approx_q'}

def run_custom_episode(game, pacman, ghosts, rules, is_eval=False, agent_type='dqn', max_steps=1000):
    """
    Custom game loop that yields transitions so we can call .update() on our agents.
    max_steps caps episode length to prevent infinite loops (e.g. RandomGhost never catches Pacman).
    """
    game.display.initialize(game.state.data)
    
    # Initialize agents
    for i, agent in enumerate(game.agents):
        if hasattr(agent, 'registerInitialState'):
            agent.registerInitialState(game.state.deepCopy())
            
    agentIndex = game.startingIndex
    numAgents = len(game.agents)
    
    # Track states and actions for updates
    last_states = {i: None for i in range(numAgents)}
    last_actions = {i: None for i in range(numAgents)}
    last_scores = {i: game.state.getScore() for i in range(numAgents)}
    
    episode_loss = 0
    updates_count = 0
    entropy_sum = 0
    step_count = 0
    
    while not game.gameOver:
        agent = game.agents[agentIndex]
        state = game.state.deepCopy()
        
        # Reward for the agent since its last turn.
        # Deep Pacman agents use dense shaped rewards for richer learning signal.
        # All other agents (and ghosts) use raw score delta.
        if agentIndex == 0 and uses_shaped_pacman_reward(agent_type) and last_states[agentIndex] is not None:
            reward = shape_pacman_reward(last_states[agentIndex], state)
        else:
            reward = state.getScore() - last_scores[agentIndex]
            if agentIndex > 0:
                reward = -reward  # Ghosts want lower score
            
        done = state.isWin() or state.isLose()
        
        # Perform update for the transition from this agent's previous state
        if last_states[agentIndex] is not None and hasattr(agent, 'update'):
            loss = agent.update(last_states[agentIndex], last_actions[agentIndex], state, reward, done)
            if loss is not None:
                episode_loss += loss
                updates_count += 1
                
        if done:
            break
            
        action = agent.getAction(state)
        game.moveHistory.append((agentIndex, action))
        
        last_states[agentIndex] = state
        last_actions[agentIndex] = action
        last_scores[agentIndex] = state.getScore()
        
        # Step environment
        game.state = game.state.generateSuccessor(agentIndex, action)
        game.display.update(game.state.data)
        rules.process(game.state, game)
        
        agentIndex = (agentIndex + 1) % numAgents
        step_count += 1
        if step_count >= max_steps:
            break  # Prevent infinite episodes (e.g. RandomGhost never catches Pacman)

    # Final updates for all agents
    for i, agent in enumerate(game.agents):
        if i == 0 and uses_shaped_pacman_reward(agent_type) and last_states[i] is not None:
            reward = shape_pacman_reward(last_states[i], game.state.deepCopy())
        else:
            reward = game.state.getScore() - last_scores[i]
            if i > 0:
                reward = -reward
        done = True
        
        if last_states[i] is not None and hasattr(agent, 'update'):
            loss = agent.update(last_states[i], last_actions[i], game.state.deepCopy(), reward, done)
            if loss is not None:
                episode_loss += loss
                updates_count += 1
                
        if hasattr(agent, 'final'):
            final_result = agent.final(game.state.deepCopy())
            if isinstance(final_result, tuple):
                final_loss, final_entropy = final_result
                if final_loss is not None:
                    episode_loss += final_loss
                    updates_count += 1
                if final_entropy is not None:
                    entropy_sum += final_entropy
            
    game.display.finish()
    
    final_score = game.state.getScore()
    is_win = game.state.isWin()
    avg_loss = episode_loss / max(1, updates_count) if updates_count > 0 else None
    
    avg_entropy = entropy_sum / max(1, updates_count) if entropy_sum > 0 else None
    
    return final_score, is_win, avg_loss, avg_entropy

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--agent', type=str, default='dqn',
                        help="q_learning, sarsa, approx_q, dqn, reinforce, ppo")
    parser.add_argument('--episodes', type=int, default=None,
                        help="Override num_episodes from config")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.episodes is not None:
        config['training']['num_episodes'] = args.episodes
    
    logger = Logger(config['training']['results_dir'])
    
    # Environment Setup
    lay = layout.getLayout(config['env']['layout'])
    rules = ClassicGameRules(timeout=config['env']['timeout'])
    display = NullGraphics() # Headless training
    rules.quiet = True
    
    pacman = create_pacman_agent(args.agent, config)
    ghosts = create_ghost_agents(config['env']['num_ghosts'], config['multiagent']['mode'], config)
    
    num_episodes = config['training']['num_episodes']
    alternating_interval = config['multiagent']['alternating_interval']
    max_steps = config['training'].get('max_steps_per_episode', 1000)
    plot_interval = config['training'].get('plot_interval', config['training']['save_interval'])
    
    start_time = datetime.now()
    start_clock = perf_counter()
    print(f"Starting training for {num_episodes} episodes with agent {args.agent}")
    print(f"Run started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if hasattr(pacman, 'device'):
        print(f"Training device: {pacman.device}")

    best_win_rate = -1.0
    best_avg_reward = -float('inf')
    
    for ep in range(num_episodes):
        # Handle Alternating Training
        if config['multiagent']['mode'] == "rl_q":
            if (ep // alternating_interval) % 2 == 0:
                pacman.is_eval = False
                for g in ghosts: g.set_learning(False)
            else:
                pacman.is_eval = True
                for g in ghosts: g.set_learning(True)
                
        game = rules.newGame(lay, pacman, ghosts, display, quiet=True, catchExceptions=False)
        
        score, is_win, loss, entropy = run_custom_episode(
            game, pacman, ghosts, rules, agent_type=args.agent, max_steps=max_steps
        )
        
        # We record episode steps by getting agent move history length
        length = len(game.moveHistory)
        logger.log_episode(score, length, is_win, loss, entropy)
        
        if (ep + 1) % 10 == 0:
            avg_reward = sum(logger.metrics['episode_rewards'][-10:]) / 10
            win_rate = logger.metrics['win_rates'][-1]
            eps = get_exploration_value(pacman)
            entropy_value = get_entropy_value(pacman, entropy)
            entropy_coef = get_entropy_coef(pacman)
            learning_rate = get_learning_rate(pacman)
            eps_msg = f" | Eps: {eps:.2f}" if eps is not None else ""
            entropy_msg = f" | Entropy: {entropy_value:.2f}" if entropy_value is not None else ""
            entropy_coef_msg = f" | EntCoef: {entropy_coef:.3f}" if entropy_coef is not None else ""
            lr_msg = f" | LR: {learning_rate:.5f}" if learning_rate is not None else ""
            print(
                f"Episode {ep+1}/{num_episodes} | "
                f"Avg Reward (last 10): {avg_reward:.1f} | "
                f"Win Rate: {win_rate:.2f}"
                f"{eps_msg}"
                f"{entropy_msg}"
                f"{entropy_coef_msg}"
                f"{lr_msg}"
            )

            is_better = (
                win_rate > best_win_rate
                or (win_rate == best_win_rate and avg_reward > best_avg_reward)
            )
            if is_better and hasattr(pacman, 'save'):
                best_win_rate = win_rate
                best_avg_reward = avg_reward
                best_path = os.path.join(config['training']['results_dir'], f"{args.agent}_best.pt")
                pacman.save(best_path)
            
        if (ep + 1) % config['training']['save_interval'] == 0:
            logger.save_logs(prefix=args.agent)
            if (ep + 1) % plot_interval == 0:
                logger.plot_metrics(prefix=args.agent)
            
            # Save the model
            if hasattr(pacman, 'save'):
                model_path = os.path.join(config['training']['results_dir'], f"{args.agent}_final.pt")
                pacman.save(model_path)
            
    end_time = datetime.now()
    elapsed_seconds = perf_counter() - start_clock
    elapsed_minutes, seconds = divmod(elapsed_seconds, 60)
    hours, minutes = divmod(int(elapsed_minutes), 60)

    print("Training Complete. Results saved.")
    print(f"Run ended at:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime:  {hours:02d}:{minutes:02d}:{seconds:05.2f} (HH:MM:SS)")

if __name__ == '__main__':
    train()
