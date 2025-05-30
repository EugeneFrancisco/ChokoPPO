import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import config
import random

class QAgent(nn.Module):
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
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_actions)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.Q_LEARNING_RATE)
        

        self.critic.to(self.device)

    def switch_to_cpu(self):
        if self.device.type == 'cpu':
            pass
        self.critic.cpu()

    def switch_to_device(self):
        if self.device.type != 'cpu':
            pass
        self.critic.to(self.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)
    
    def sample_action(self, obs: torch.Tensor, mask: torch.Tensor, eps_greedy = True) -> torch.Tensor:
        '''
        Given an observation and a mask of the action space, samples an action from the policy implied
        by the critic. Defaults to epsilon greedy sampling, with 
        '''

        q_values = self.critic(obs.unsqueeze(0)).squeeze(0) 
        mask = mask.squeeze(0) # [1, 2625] -> [2625]
        masked_q = q_values + (1.0 - mask) * config.NEG_INF


        if eps_greedy and random.random() < self.epsilon:
            valid_idxs = torch.nonzero(mask, as_tuple=False)  # e.g. tensor([2, 5, 7])
            rand_pos = random.randint(0, valid_idxs.shape[0] - 1)
            action = valid_idxs[rand_pos]
        else:
            action = torch.argmax(masked_q)

        return action 
    
    def get_value(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        '''
        Given an observation and an action, returns the value of the action.
        '''
        # actions is dim [batch_size, 1]
        # TODO, this is fucked does not work fix it later
        all_q_values = self.critic(obs) # [batch_size, num_actions]
        num_actions = all_q_values.shape[0]
        q_values = all_q_values[torch.arange(0, num_actions - 1), actions]
        return q_values

    def act(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q_values = self.critic(obs).squeeze(0)
        masked_q = (q_values + (1.0 - mask.float()) * config.NEG_INF).squeeze(0)
        max_index = torch.argmax(masked_q)
        probs = torch.zeros((self.num_actions,))
        probs[max_index] = 1.0
        dist = Categorical(probs)
        return dist
   
