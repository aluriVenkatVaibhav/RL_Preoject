import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNet(nn.Module):
    """
    Dueling DQN network.
    Architecture: 3 conv layers → shared FC → separate Value + Advantage streams.
    Dueling decomposition: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
    """
    def __init__(self, input_shape, num_actions):
        super(DQNNet, self).__init__()
        c, h, w = input_shape

        # ── Convolutional feature extractor ──────────────────────────────────
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Compute flattened conv output size (stride=1, padding=1 preserves H×W)
        conv_out_size = 64 * h * w

        # ── Shared fully-connected layer ──────────────────────────────────────
        self.fc_shared = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
        )

        # ── Dueling streams ───────────────────────────────────────────────────
        # Value stream: V(s) — scalar
        self.value_stream = nn.Linear(512, 1)

        # Advantage stream: A(s, a) — per-action
        self.advantage_stream = nn.Linear(512, num_actions)

        # ── Weight initialisation (He / Kaiming for ReLU layers) ──────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Convolutional features
        x = self.conv(x)
        x = x.view(x.size(0), -1)          # Flatten

        # Shared layer
        x = self.fc_shared(x)

        # Dueling decomposition
        value = self.value_stream(x)         # (B, 1)
        advantage = self.advantage_stream(x) # (B, num_actions)

        # Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
