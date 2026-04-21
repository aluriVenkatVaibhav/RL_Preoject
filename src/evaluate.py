import argparse
import os
import sys

# Parse args first so we can resolve paths against the ORIGINAL cwd
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--config', type=str, default='config.yaml')
_pre_parser.add_argument('--model_path', type=str, default='')
_pre_args, _ = _pre_parser.parse_known_args()
_ABS_CONFIG = os.path.abspath(_pre_args.config)
_ABS_MODEL  = os.path.abspath(_pre_args.model_path) if _pre_args.model_path else ''

# Now switch to src/ so all game imports and layout lookups work
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from pacman import ClassicGameRules
import layout
from utils.logger import Logger
import textDisplay
import graphicsDisplay

from train import load_config, create_pacman_agent, create_ghost_agents, run_custom_episode

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--agent', type=str, default='dqn', help="q_learning, sarsa, approx_q, dqn, reinforce, ppo")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model file")
    parser.add_argument('--episodes', type=int, default=10, help="Number of games to evaluate")
    parser.add_argument('--headless', action='store_true', help="Run without graphics")
    args = parser.parse_args()
    
    # Use the pre-resolved absolute paths (resolved before chdir ran)
    config = load_config(_ABS_CONFIG)
    model_path = _ABS_MODEL
    
    lay = layout.getLayout(config['env']['layout'])
    if lay is None:
        print(f"ERROR: Layout '{config['env']['layout']}' not found. Check config.yaml.")
        return
    rules = ClassicGameRules(timeout=config['env']['timeout'])
    
    if args.headless:
        display = textDisplay.NullGraphics()
        rules.quiet = True
    else:
        display = graphicsDisplay.PacmanGraphics(1.0, frameTime=0.1)
        rules.quiet = False

    pacman = create_pacman_agent(args.agent, config)
    pacman.is_eval = True
    
    # Needs to be called with a state to initialize network input shapes if using dl agent
    # We load standard ghosts
    ghosts = create_ghost_agents(config['env']['num_ghosts'], 'baseline', config)
    
    # We must start a dummy game to let initialize network trigger before load
    test_game = rules.newGame(lay, pacman, ghosts, textDisplay.NullGraphics(), quiet=True, catchExceptions=False)
    for agent in test_game.agents:
        if hasattr(agent, 'registerInitialState'):
            agent.registerInitialState(test_game.state.deepCopy())
    
    # Load
    if hasattr(pacman, 'load') and model_path:
        pacman.load(model_path)
    
    scores = []
    wins = []
    
    print(f"Starting evaluation of {args.agent} for {args.episodes} episodes...")
    
    for i in range(args.episodes):
        game = rules.newGame(lay, pacman, ghosts, display, quiet=args.headless, catchExceptions=False)
        score, is_win, _, _ = run_custom_episode(
            game,
            pacman,
            ghosts,
            rules,
            is_eval=True,
            agent_type=args.agent,
            max_steps=config['training'].get('max_steps_per_episode', 1000),
        )
        scores.append(score)
        wins.append(is_win)
        
        print(f"Eval Episode {i+1} - Score: {score}, Win: {is_win}")
        
    avg_score = sum(scores) / len(scores)
    win_rate = sum(wins) / len(wins)
    
    print("\n--- Evaluation Results ---")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Win Rate:      {win_rate*100:.2f}%")

if __name__ == '__main__':
    evaluate()
