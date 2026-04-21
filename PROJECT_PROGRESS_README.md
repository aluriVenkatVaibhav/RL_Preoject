# Pacman RL Project Progress README

This file documents the current state of the Pacman reinforcement learning project,
the fixes made so far, and the direction for the remaining agents. It is separate
from the original `README.md` so the project overview stays intact while experiment
notes can evolve.

## Project Summary

This project extends the Berkeley Pacman environment into a reinforcement learning
experimentation framework. It supports flat RL agents, deep RL agents, hierarchical
Pacman agents, RL ghost agents, shaped rewards, training logs, saved models, and
plots.

The main training entry point is:

```powershell
cd D:\IIITDM\Sem-6\RL\Project\RL_Project\src
..\venv\Scripts\Activate.ps1
python train.py --config <config_file> --agent <agent_name> --episodes <num_episodes>
```

Available flat agents include:

- `q_learning`
- `sarsa`
- `approx_q`
- `dqn`
- `reinforce`
- `ppo`

The main code areas are:

- `src/train.py`: common training loop for flat agents
- `src/evaluate.py`: evaluation runner for saved models
- `src/agents/`: agent implementations
- `src/models/`: neural network models
- `src/utils/`: state parsing, reward shaping, replay buffer, logging
- `src/results/`: saved models, metrics, and plots
- `src/layouts/`: Pacman map layouts

## Important Config Files

### `src/config.yaml`

General project config. It currently uses `mediumClassic` with two ghosts. This is
harder and slower than the small DQN/PPO experiment setup.

### `src/dqn_config.yaml`

DQN-focused config for the improved baseline. It uses:

- `smallClassic`
- one random ghost
- 500 max steps per episode
- slower/stable DQN learning rate
- DQN epsilon decay by training steps
- reward scaling/clipping
- no `Stop` action when movement is available

Recommended DQN command:

```powershell
python train.py --config dqn_config.yaml --agent dqn --episodes 2000
```

### `src/ppo_config.yaml`

PPO-focused config created to avoid confusion with `dqn_config.yaml`. It keeps the
same `smallClassic` environment for fair comparison against DQN, but names the file
for PPO experiments.

Recommended PPO command:

```powershell
python train.py --config ppo_config.yaml --agent ppo --episodes 2000
```

Using `dqn_config.yaml` for PPO will not damage DQN results because model and log
outputs are prefixed by agent name, but `ppo_config.yaml` is cleaner.

## Output Safety

Training saves outputs using the selected agent name:

- DQN model: `src/results/dqn_final.pt`
- PPO model: `src/results/ppo_final.pt`
- DQN logs: `src/results/logs/dqn_metrics.*`
- PPO logs: `src/results/logs/ppo_metrics.*`
- DQN plots: `src/results/plots/dqn_*.png`
- PPO plots: `src/results/plots/ppo_*.png`

So running:

```powershell
python train.py --config ppo_config.yaml --agent ppo --episodes 2000
```

does not overwrite the DQN model or DQN metrics.

## DQN Work Completed

The DQN agent was the first target because it was failing to learn reliably after
2000 episodes. The old behavior showed mostly negative rewards and weak/no win-rate
improvement.

Key changes made:

- Added Double DQN target selection/evaluation.
- Added soft target-network updates with Polyak averaging.
- Added warmup before optimization.
- Added step-based epsilon decay from config.
- Added gradient norm clipping.
- Replaced the DQN model with a dueling DQN architecture.
- Added legal-action masking to replay targets so DQN no longer bootstraps from
  impossible next actions.
- Added optional `Stop` avoidance when movement is available.
- Added reward scaling/clipping before DQN optimization.
- Added dense shaped reward for DQN Pacman transitions.
- Added terminal shaped reward consistency in the final update.
- Added max-steps-per-episode support to avoid stuck runs.

Important DQN files:

- `src/agents/dqn_agent.py`
- `src/models/dqn_net.py`
- `src/utils/replay_buffer.py`
- `src/utils/reward_shaper.py`
- `src/utils/state_parser.py`
- `src/train.py`
- `src/dqn_config.yaml`

## DQN Result Inference

The latest DQN run showed strong learning:

- Early training: rewards were negative and win rate was zero.
- Around episode 230-320: rewards started becoming positive.
- Around episode 330 onward: wins began appearing.
- Around episode 900-1600: trailing win rate reached roughly 0.8 to 0.93.
- Final episode 2000: trailing win rate was around 0.86 with strongly positive reward.

Conclusion: DQN is now a working baseline. It should be protected while improving
other agents.

Future DQN improvements, if needed:

- Save the best checkpoint by trailing win rate, not only the final checkpoint.
- Add deterministic evaluation checkpoints during training.
- Log epsilon, loss, score, win rate, and runtime into CSV together.

## Training Loop Improvements Completed

The common training loop now prints:

- run start time
- run end time
- total runtime
- epsilon/exploration value when the agent exposes it

Example output:

```text
Starting training for 2000 episodes with agent dqn
Run started at: 2026-04-21 02:53:38
Episode 10/2000 | Avg Reward (last 10): -412.7 | Win Rate: 0.00 | Eps: 0.99
...
Training Complete. Results saved.
Run ended at:   2026-04-21 03:42:10
Total runtime:  00:48:32.12 (HH:MM:SS)
```

## PPO Work Started

PPO was selected as the next agent because it is the next deep RL baseline and is
more likely than tabular agents to suffer from scale, variance, and policy-update
issues.

Changes made so far:

- Added PPO-specific reward scaling and clipping.
- Added PPO-specific `Stop` avoidance.
- Avoided bootstrapping terminal states with a learned critic value.
- Switched critic loss from MSE to Huber loss for more stable value learning.
- Reset PPO trajectory bookkeeping after each update.
- Added `src/ppo_config.yaml`.
- Added shaped Pacman rewards for PPO in the training loop.
- Added PPO-specific learning rate through `ppo_learning_rate`.
- Changed PPO action distributions to use masked logits directly.
- Added PPO loss and entropy reporting from the final episode update.
- Added entropy printing in the training trace as PPO's exploration diagnostic.
- Changed PPO to update from 5-episode rollout batches instead of one episode at
  a time.
- Increased PPO entropy pressure in `ppo_config.yaml` to reduce premature policy
  collapse.

Important PPO files:

- `src/agents/ppo_agent.py`
- `src/models/actor_critic_net.py`
- `src/ppo_config.yaml`

PPO smoke test completed:

```powershell
python train.py --config dqn_config.yaml --agent ppo --episodes 5
```

The run completed successfully and printed runtime information.

Initial full PPO run before the latest fixes did not learn:

- rewards stayed around -400 to -500
- win rate stayed at 0.00 through 2000 episodes

Diagnosis:

- PPO was receiving raw score deltas while DQN received dense shaped rewards.
- PPO inherited the conservative DQN learning rate unless separately configured.
- PPO had no exploration diagnostic in the console.

After the fixes, a 200-episode PPO check showed reward improvement from roughly
`-415` early to around `-230` near episode 200. This is not a solved PPO model yet,
but it is a clear improvement over the failed flat trace.

A later 2000-episode PPO run improved much more than the original failed run, but
still remained weak:

- rewards gradually improved from around `-400` toward occasional near-zero or
  positive windows
- first wins appeared very late, around episode 1570
- win rate stayed low and unstable
- entropy sometimes collapsed near zero, meaning the policy became too
  deterministic too early

Follow-up PPO changes were added after that run:

- PPO now updates every 5 episodes to reduce variance.
- PPO entropy coefficient in `ppo_config.yaml` is now `0.06`.
- Entropy output is reused between update episodes so exploration is visible in
  the console trace.
- PPO now has a faster entropy schedule: `0.05` down to `0.003` over 150 update
  batches.
- PPO uses `ppo_batch_size: 128` to make neural updates more GPU-friendly.
- The actor-critic network now uses a DQN-sized CNN backbone.
- Training now saves `<agent>_best.pt` separately from `<agent>_final.pt` when
  trailing win rate/reward improves.
- PPO now uses clipped value updates (`ppo_value_clip`) and a PPO-only learning
  rate decay schedule.
- Training output now prints PPO learning rate in addition to entropy and entropy
  coefficient.

Latest 200-episode check after batched updates:

- entropy stayed healthier, roughly `0.6-0.8`
- rewards improved in the second half, reaching around `-240` to `-290`
- this is more stable, though not solved yet

After the actor-critic and entropy-schedule update, a 300-episode probe showed:

- CUDA was active: `Training device: cuda`
- entropy coefficient decayed from `0.050` to about `0.032`
- rewards improved into the `-150` to `-190` range by the end of the probe
- wins still did not appear in 300 episodes, so full-run behavior must be checked

Latest follow-up changes after the weak 2000-episode PPO run:

- Added `ppo_learning_rate_end: 0.00008`
- Added `ppo_lr_decay_updates: 150`
- Added `ppo_value_clip: 0.2`

These are intended to reduce late-training oscillation without touching DQN.

Recommended next PPO test:

```powershell
python train.py --config ppo_config.yaml --agent ppo --episodes 2000
```

## DQN Protection Rule

While improving other agents, avoid changing these unless explicitly needed:

- `src/agents/dqn_agent.py`
- `src/models/dqn_net.py`
- `src/dqn_config.yaml`
- DQN-specific reward handling in `src/train.py`

Shared utility changes should be backward-compatible and should not alter DQN
training semantics.

## Suggested Roadmap

1. Run PPO for 2000 episodes using `ppo_config.yaml`.
2. Compare PPO reward and win-rate curves against DQN.
3. If PPO underperforms, tune PPO-specific values only:
   - `ppo_reward_scale`
   - `ppo_reward_clip`
   - `entropy_coef`
   - `ppo_epochs`
   - `gae_lambda`
4. Improve REINFORCE next, because it shares the actor-critic network and can reuse
   some PPO stability lessons.
5. Improve tabular Q-learning/SARSA after deep agents.
6. Return to hierarchical agents after flat baselines are stable.

## Current Recommended Commands

DQN:

```powershell
cd D:\IIITDM\Sem-6\RL\Project\RL_Project\src
..\venv\Scripts\Activate.ps1
python train.py --config dqn_config.yaml --agent dqn --episodes 2000
```

PPO:

```powershell
cd D:\IIITDM\Sem-6\RL\Project\RL_Project\src
..\venv\Scripts\Activate.ps1
python train.py --config ppo_config.yaml --agent ppo --episodes 2000
```

Evaluate DQN:

```powershell
python evaluate.py --config dqn_config.yaml --agent dqn --model_path results/dqn_final.pt --episodes 10 --headless
```

Evaluate PPO:

```powershell
python evaluate.py --config ppo_config.yaml --agent ppo --model_path results/ppo_final.pt --episodes 10 --headless
```

## Notes

The DQN improvement is significant and should be treated as the current baseline.
PPO is now prepared for a serious 2000-episode run, but it has not yet been proven
the way DQN has. Its results should be evaluated from logs and plots after the full
run.

For runtime, `plot_interval` was added to the config files. Models/logs still save
every `save_interval`, but plots are generated less often. This should not affect
learning quality because plotting is only reporting overhead.

PyTorch CUDA was verified in the project venv:

```text
torch 2.7.1+cu118
cuda_available True
device NVIDIA GeForce RTX 4060 Laptop GPU
```

The neural networks run on GPU, but the Berkeley Pacman environment remains mostly
CPU/Python-bound. Fully making training GPU-bound would require vectorizing or
rewriting the environment, not just moving the model to CUDA.
