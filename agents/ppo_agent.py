import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import config

class PPOAgent(nn.Module):
    def __init__(self, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.trunk = nn.Sequential(
            nn.Linear(25, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(self.hidden_dim, self.num_actions)
        self.critic_head = nn.Linear(self.hidden_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE)
        self.trunk.to(self.device)
        self.actor_head.to(self.device)
        self.critic_head.to(self.device)
    
    def switch_to_cpu(self):
        '''
        Switches the model to CPU.
        '''
        if self.device.type == 'cpu':
            pass
        self.trunk.cpu()
        self.actor_head.cpu()
        self.critic_head.cpu()
    
    def switch_to_device(self):
        if self.device.type != 'cpu':
            pass
        self.trunk.to(self.device)
        self.actor_head.to(self.device)
        self.critic_head.to(self.device)


    
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk_out = self.trunk(obs)
        action_logits = self.actor_head(trunk_out)
        value = self.critic_head(trunk_out)
        return action_logits, value
    
    def act(self, obs: torch.Tensor, mask: torch.Tensor) -> Categorical:
        '''
        Given an observation and a mask of the action space,
        returns an action sampled from the policy.
        Args:
            obs: [1, 25] tensor of observations
            mask: [1, 2625] tensor of action mask (note that 1 means keep and 0 means remove)
        Returns:
            action: int, the action sampled from the policy
        '''
        
        trunk_out = self.trunk(obs)
        action_logits = self.actor_head(trunk_out)
        masked_logits = action_logits + (1 - mask) * config.NEG_INF
        probs = torch.softmax(masked_logits, dim=-1)
        dist = Categorical(probs)
        return dist
    
    def act_greedy(self, obs: torch.Tensor, mask: torch.Tensor) -> int:
        '''
        Given an observation and a mask of the action space,
        returns the action sampled from the policy.
        Args:
            obs: [1, 25] tensor of observations
            mask: [1, 2625] tensor of action mask (note that 1 means keep and 0 means remove)
        Returns:
            action: int, the action sampled from the policy
        '''
        trunk_out = self.trunk(obs)
        action_logits = self.actor_head(trunk_out)
        masked_logits = action_logits + (1 - mask) * config.NEG_INF
        probs = torch.softmax(masked_logits, dim=-1)
        action = torch.argmax(probs).item()
        return action
    
    def critic(self, obs: torch.Tensor) -> torch.Tensor:
        '''
        Given a batch of observations, returns a batch of values.
        Args:
            obs: [batch_size, 25] tensor of observations
        Returns:
            value: [batch_size, 1] tensor of values
        '''
        trunk_out = self.trunk(obs)
        value = self.critic_head(trunk_out)
        return value.squeeze(1)