import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import config

class PPOAgent(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.hidden_dim = hidden_dim
        self.num_actions = config.NUM_ACTIONS
        self.epsilon = config.EPSILON
        self.critic = nn.Sequential( # TODO, more layers here
            nn.Linear(25, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_actions)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE)
        

        self.critic.to(self.device)

    def switch_to_cpu(self):
        # TODO
        pass

    def switch_to_device(self):
        # TODO
        pass

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)
    
    def sample_action(self, obs: torch.Tensor, mask: torch.Tensor, eps_greedy = True) -> torch.Tensor:
        '''
        Given an observation and a mask of the action space, samples an action from the policy implied
        by the critic. Defaults to epsilon greedy sampling, with 
        '''
        q_values = self.critic(obs)
        masked_q_values = q_values + (1 - mask) * config.NEG_INF
        if eps_greedy:
            # Epsilon greedy sampling
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.num_actions, p = mask.cpu().numpy())
            else:
                action = torch.argmax(masked_q_values).item()
        else:
            action = torch.argmax(masked_q_values).item()
        
        return action
