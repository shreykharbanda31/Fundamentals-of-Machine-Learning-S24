"""Critic (value) deep neural network.

Author: Elie KADOCHE.
"""

import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

class CriticModel(nn.Module):

    #By default use CPU
    DEVICE = torch.device("cpu")

    def __init__(self, input_size):
        """Initialize model"""
        super(CriticModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.crit = nn.Linear(128, 1)

    def _preprocessor(self, state):
        """Preprocessor function.

        Args:
            state (numpy.array): environment state.

        Returns:
            x (torch.tensor): preprocessed state.
        """
        # Add batch dimension
        x = np.expand_dims(state, 0)

        # Transform to torch.tensor
        x = torch.from_numpy(x).float().to(self.DEVICE)

        return x

    def forward(self, state):
        """Forward pass.

        Args:
            state (numpy.array): environment state.

        Returns:
            state_value: list with values of estimated returns
        """

        # Preprocessor
        x = self._preprocessor(state)

        # Input layer
        x = F.relu(self.fc1(x))

        # Hidden Layer
        x = self.fc2(x)

        # Critic
        state_value = F.softmax(self.crit(x), dim=-1)
        
        return state_value
