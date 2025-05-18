import torch
import numpy as np

NUM_ACTIONS = (25) + (25 * 4) + (25 * 4 * 25) # 2625 possible actions
class RandomAgent:
    def __init__(self):
        self.num_actions = NUM_ACTIONS

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            obs: [batch_size, 25] tensor of observations
        Returns:
            action: [batch_size, num_actions] tensor of actions
        '''
        return torch.rand(obs.shape[0], self.num_actions)

