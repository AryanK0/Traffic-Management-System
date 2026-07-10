"""
=========================================================
DQN Network
=========================================================

The exact network architecture that was trained in SUMO.
fc1 (6 -> 64) -> fc2 (64 -> 64) -> fc3 (64 -> 2)

State (6):
    [0-3] n/s/e/w incoming queue counts / 100
    [4]   junction spillback count / 100
    [5]   time in current phase / 60

Actions (2):
    0 = KEEP current phase
    1 = SWITCH phase

Author : Traffic AI Team
Platform : NVIDIA Jetson Nano
=========================================================
"""

import torch.nn as nn
import torch.nn.functional as F

import config


class DQN(nn.Module):

    def __init__(self,
                 input_size=config.STATE_SIZE,
                 hidden_size=64,
                 output_size=config.ACTION_SIZE):

        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)
