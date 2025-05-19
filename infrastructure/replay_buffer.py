from envs.choko_env import Choko_Env
from agents.random_agent import RandomAgent
from torch.utils.data import Dataset, DataLoader
from . import utils
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
    def __init__(self, gamma, lam, max_size = MAX_SIZE):
        self.gamma = gamma
        self.lam = lam
        self.max_size = max_size
        self.env = Choko_Env()
        self.agent = RandomAgent()
        self.critic = lambda x: 10 # dummy critic TODO

        pass
    def make_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        dataset_obs = []
        actions = []
        advantages = []
        returns = []

        while len(dataset_obs) < self.max_size:
            # sample a trajectory
            player_1_states = []
            player_1_actions = []
            player_1_rewards = []
            player_1_values = []
            
            player_2_states = []
            player_2_actions = []
            player_2_rewards = []
            player_2_values = []

            raw_obs, mask = self.env.reset()
            with torch.no_grad():
                while True:
                    # TODO make sure choko returns the observation with flipped 1s and 2s
                    # (1) add the observation
                    if self.env.player == 1:
                        obs = raw_obs
                        obs_torch = torch.from_numpy(raw_obs).float().unsqueeze(0)
                        player_1_states.append(obs)
                        #TODO, leave value generation for the end so that you can do one batch pass over the states
                        player_1_values.append(self.critic(obs_torch))
                    else:
                        obs = 3 - raw_obs
                        obs_torch = 3 - torch.from_numpy(raw_obs).float().unsqueeze(0)
                        player_2_states.append(obs)
                        player_2_values.append(self.critic(obs_torch))
                    # (2) get the action and add the action
                    logits = self.agent.forward(obs_torch) # note obs here is just one observation TODO, maybe change to torch?
                    mask = torch.from_numpy(mask).float().unsqueeze(0)
                    assert logits.shape == mask.shape
                    masked_logits = logits + (1 - mask) * NEG_INF
                    probs = torch.softmax(masked_logits, dim = -1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()

                    if self.env.player == 1:
                        player_1_actions.append(action.item())
                    else:
                        player_2_actions.append(action.item())

                    state, reward, done, _ = self.env.step(action.item())
                    raw_obs, mask = state

                    # (3) add the reward (careful because player just switched after step)
                    if self.env.player == 2:
                        player_1_rewards.append(reward)
                    else:
                        player_2_rewards.append(reward)
                    
                    if done != "ongoing":
                        # the game is over, need to check if this player won because then the
                        # last reward for the opponent is -1.
                        if reward == 1:
                            # the player who just played won
                            if self.env.player == 2:
                                # player 1 won, need to set the last reward for player 2 to -1
                                player_2_rewards[-1] = -1
                            else:
                                # player 2 won, need to set the last reward for player 1 to -1
                                player_1_rewards[-1] = -1
                        # add the last state to the trajectory
                        player_1_values.append(0.0)
                        player_2_values.append(0.0)
                        break

                # now we have trajectories from each player's perspective. Need to add them to
                # experiences
                player_1_returns, player_1_advantages = utils.compute_gae_and_returns(
                    player_1_rewards,
                    player_1_values,
                    self.gamma,
                    self.lam
                )

                dataset_obs.extend(player_1_states)
                actions.extend(player_1_actions)
                advantages.extend(player_1_advantages)
                returns.extend(player_1_returns)

                player_2_returns, player_2_advantages = utils.compute_gae_and_returns(
                    player_2_rewards,
                    player_2_values,
                    self.gamma,
                    self.lam
                )

                dataset_obs.extend(player_2_states)
                actions.extend(player_2_actions)
                advantages.extend(player_2_advantages)
                returns.extend(player_2_returns)
            
        # convert the lists to numpy arrays
        dataset_obs = np.array(dataset_obs)
        actions = np.array(actions)
        advantages = np.array(advantages)
        returns = np.array(returns)

        return dataset_obs, actions, advantages, returns



if __name__ == "__main__":
    env = Choko_Env()
    buffer = ReplayBuffer(0.99, 0.95, 64)
    dataset_obs, actions, advantages, returns = buffer.make_dataset()
    print(dataset_obs.shape)


