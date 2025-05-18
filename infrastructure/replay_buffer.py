from envs.choko_env import Choko_Env
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

MAX_SIZE = 1000000

class RLDataset(Dataset):
    def __init__(self, S, A, R, S_next, D):
        self.S = torch.from_numpy(S)
        self.A = torch.from_numpy(A)
        self.R = torch.from_numpy(R)
        self.S_next = torch.from_numpy(S_next)
        self.D = torch.from_numpy(D)
    
    def __len__(self):
        return len(self.S)
    
    def __getitem__(self, idx):
        return (self.S[idx], self.A[idx], self.R[idx], self.S_next[idx], self.D[idx])

class ReplayBuffer:
    # create a replay buffer that stores the MAX_SIZE transitions
    # i.e., there are MAX_SIZE tuples of (s, a, r, s', d) in the buffer
    def __init__(self, max_size = MAX_SIZE):
        self.max_size = max_size
        self.env = Choko_Env()
        
        pass
    def make_dataset(self):
        # create an empty list of rollouts
        #
        # populate the list with rollouts, each element of the list is one rollout.
        # each rollout is a list of tuples (st, at, rt, v(st), s_{t + 1}, ...)
        # 
        # for each rollout (trajectory) in the list:
        #   perform the backward recursion to make one
        #   transition tuple (s, a, At, Rt) for each time step t in the rollout.
        #
        # combine all the transition tuples across all the rollouts together to make a dataset.
        pass



if __name__ == "__main__":
    env = Choko_Env()

