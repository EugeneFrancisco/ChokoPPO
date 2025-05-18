from envs.choko_env import Choko_Env
from agents.random_agent import RandomAgent
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

MAX_SIZE = 1000000
NEG_INF = -1e10

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
        self.agent = RandomAgent()
        self.critic = lambda x: 10 # dummy critic TODO

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
        experiences = []
        lengths = [] #stores the lengths of each trajectory to differentiate between episodes
        while len(experiences) < self.max_size:
            # sample a trajectory
            player_1_trace = []
            player_2_trace = []

            obs, mask = self.env.reset()
            while True:
                # first, player 1's turn
                logits = self.agent.forward(obs) # note obs here is just one observation
                masked_logits = logits + (1 - mask) * NEG_INF
                probs = torch.softmax(masked_logits, dim = -1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

                state, reward, done, _ = self.env.step(action)
                obs, mask = state

                if done:
                    # the game is over, need to check if this player won because then the
                    # last reward for the opponent is -1.
                    if reward == 1:
                        # the player who just played won
                        if self.env.player == 2:
                            # player 1 won, need to set the last reward for player 2 to -1
                            player_2_trace[-1] = (
                                player_2_trace[-1][0], 
                                player_2_trace[-1][1], 
                                -1, 
                                player_2_trace[-1][3]
                                )
                        else:
                            # player 2 won, need to set the last reward for player 1 to -1
                            player_1_trace[-1] = (
                                player_1_trace[-1][0], 
                                player_1_trace[-1][1], 
                                -1, 
                                player_1_trace[-1][3]
                                )
                    break

                if self.env.player == 2:
                    # player 1 just played
                    player_1_trace.append((obs, action, reward, self.critic(obs)))
                
                if self.env.player == 1:
                    # player 2 just played
                    player_2_trace.append((obs, action, reward, self.critic(obs)))
            # now we have trajectories from each player's perspective. Need to add them to
            # experiences
            # TODO, calculate the GAE returns and stuff here now that we have one experience. Can get rid of the
            # lengths list if we do that.
            experiences.extend(player_1_trace)
            experiences.extend(player_2_trace)
            lengths.append(len(player_1_trace))
            lengths.append(len(player_2_trace))
            

        pass



if __name__ == "__main__":
    env = Choko_Env()

