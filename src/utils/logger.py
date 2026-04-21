import os
import json
import csv
import matplotlib.pyplot as plt
import numpy as np

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.plots_dir = os.path.join(log_dir, 'plots')
        self.logs_dir = os.path.join(log_dir, 'logs')
        
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'win_rates': [],
            'losses': [],
            'entropies': []
        }
        self.win_history = []
        
    def log_episode(self, reward, length, is_win, loss=None, entropy=None):
        self.metrics['episode_rewards'].append(reward)
        self.metrics['episode_lengths'].append(length)
        
        self.win_history.append(1 if is_win else 0)
        # Calculate moving average win rate over last 100 episodes
        recent_wins = self.win_history[-100:]
        win_rate = sum(recent_wins) / len(recent_wins)
        self.metrics['win_rates'].append(win_rate)
        
        if loss is not None:
            self.metrics['losses'].append(loss)
        if entropy is not None:
            self.metrics['entropies'].append(entropy)
            
    def save_logs(self, prefix="experiment"):
        # Save to JSON
        json_path = os.path.join(self.logs_dir, f"{prefix}_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
        # Save to CSV
        csv_path = os.path.join(self.logs_dir, f"{prefix}_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Reward', 'Length', 'WinRate', 'Loss', 'Entropy'])
            num_episodes = len(self.metrics['episode_rewards'])
            for i in range(num_episodes):
                loss = self.metrics['losses'][i] if i < len(self.metrics['losses']) else ""
                entropy = self.metrics['entropies'][i] if i < len(self.metrics['entropies']) else ""
                writer.writerow([
                    i+1,
                    self.metrics['episode_rewards'][i],
                    self.metrics['episode_lengths'][i],
                    self.metrics['win_rates'][i],
                    loss,
                    entropy
                ])

    def plot_metrics(self, prefix="experiment"):
        """Generates required plots."""
        # Moving average helper
        def moving_average(data, window_size=50):
            if len(data) < window_size:
                return data
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

        # 1. Reward vs Episodes
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics['episode_rewards'], label='Reward', alpha=0.3)
        plt.plot(moving_average(self.metrics['episode_rewards']), label='Moving Avg', color='blue')
        plt.title('Reward vs Episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, f"{prefix}_reward.png"))
        plt.close()

        # 2. Win rate vs Episodes
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics['win_rates'], color='green')
        plt.title('Win Rate (trailing 100) vs Episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(self.plots_dir, f"{prefix}_win_rate.png"))
        plt.close()

        # 3. Loss vs Training Steps
        if self.metrics['losses']:
            plt.figure(figsize=(10, 5))
            plt.plot(self.metrics['losses'], color='red')
            plt.title('Loss vs Updates')
            plt.xlabel('Updates')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(self.plots_dir, f"{prefix}_loss.png"))
            plt.close()

        # 4. Policy Entropy vs Episodes
        if self.metrics['entropies']:
            plt.figure(figsize=(10, 5))
            plt.plot(self.metrics['entropies'], color='purple')
            plt.title('Policy Entropy vs Updates')
            plt.xlabel('Updates')
            plt.ylabel('Entropy')
            plt.savefig(os.path.join(self.plots_dir, f"{prefix}_entropy.png"))
            plt.close()
