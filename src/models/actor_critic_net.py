import torch
import torch.nn as nn


class ActorCriticNet(nn.Module):
    """
    Shared CNN actor-critic backbone for PPO/REINFORCE.

    The policy-gradient agents need both a policy and a value estimate from the
    same visual state. This uses a DQN-sized feature extractor so PPO is not
    bottlenecked by a much smaller network than the DQN baseline.
    """

    def __init__(self, input_shape, num_actions):
        super(ActorCriticNet, self).__init__()
        c, h, w = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        conv_out_size = 64 * h * w

        self.fc_shared = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(512, num_actions)
        self.critic_head = nn.Linear(512, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc_shared(x)

        logits = self.actor_head(x)
        value = self.critic_head(x)

        return logits, value
