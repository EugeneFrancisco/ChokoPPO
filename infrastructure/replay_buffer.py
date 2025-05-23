from envs.choko_env import Choko_Env
from agents.random_agent import RandomAgent
from agents.ppo_agent import PPOAgent
from torch.utils.data import Dataset, DataLoader
from . import utils
import numpy as np
import torch
import torch.nn as nn
import concurrent.futures
import tqdm
import config


class RLDataset(Dataset):
    def __init__(self, obs, actions, masks, logps, advantages, returns):
        self.obs = torch.from_numpy(obs).float()
        self.actions = torch.from_numpy(actions)
        self.masks = torch.from_numpy(masks).long()
        self.advantages = torch.from_numpy(advantages).float()
        self.logps = torch.from_numpy(logps).float()
        self.returns = torch.from_numpy(returns).float()

    
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        return (
            self.obs[idx], 
            self.actions[idx],
            self.masks[idx],
            self.logps[idx], 
            self.advantages[idx], 
            self.returns[idx]
            )

class ReplayBuffer:
    def __init__(self, gamma, lam, ppo_agent, max_size = config.MAX_BUFFER_SIZE):
        self.gamma = gamma
        self.lam = lam
        self.max_size = max_size
        self.ppo_agent = ppo_agent
        self.agent = self.ppo_agent.act
        self.critic = self.ppo_agent.critic
    
    def make_dataloader(self, batch_size = 64, shuffle = True, num_workers = 0) -> DataLoader:
        '''
        Returns a DataLoader for the dataset
        '''
        dataset_obs, actions, masks, logps, advantages, returns = self.make_dataset()
        dataset = RLDataset(dataset_obs, actions, masks, logps, advantages, returns)
        dataloader = DataLoader(
            dataset, 
            batch_size = batch_size, 
            shuffle = shuffle, 
            num_workers = num_workers,
            drop_last = True
            )
        return dataloader

    def make_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Returns a tuple of (dataset_obs, actions, advantages, returns) where:
            dataset_obs: [max_size, 25] a numpy array of observations
            actions: [max_size] a numpy array of actions
            masks: [max_size] a numpy array of masks
            logps: [max_size] a numpy array of log probabilities
            advantages: [max_size] a numpy array of advantages
            returns: [max_size] a numpy array of returns
        Strategy:
            create an empty list of rollouts
            
            populate the list with rollouts, each element of the list is one rollout.
            each rollout is a list of tuples (st, at, rt, v(st), s_{t + 1}, ...)
            
            for each rollout (trajectory) in the list:
            perform the backward recursion to make one
            transition tuple (s, a, At, Rt) for each time step t in the rollout.
            
            combine all the transition tuples across all the rollouts together to make a dataset.
        '''
        dataset_obs = []
        actions = []
        masks = []
        logps = []
        advantages = []
        returns = []

        pbar = tqdm.tqdm(total=self.max_size, desc="Collecting rollouts", unit="samples")

        while len(dataset_obs) < self.max_size:
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.run_one_episode) for _ in range(config.NUM_TASKS)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            for result in results:
                rollout_obs, rollout_actions, rollout_masks, rollout_logps, rollout_advantages, rollout_returns = result
                
                dataset_obs.extend(rollout_obs)
                actions.extend(rollout_actions)
                logps.extend(rollout_logps)
                masks.extend(rollout_masks)
                advantages.extend(rollout_advantages)
                returns.extend(rollout_returns)
                pbar.update(len(rollout_obs))

        pbar.close()
        
        # convert the lists to numpy arrays
        dataset_obs = np.array(dataset_obs)
        actions = np.array(actions)
        masks = np.array(masks, dtype=np.int8)
        logps = np.array(logps)
        advantages = np.array(advantages)

        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8    
        advantages = (advantages - adv_mean) / adv_std

        returns = np.array(returns)

        return dataset_obs, actions, masks, logps, advantages, returns
    
    def run_one_episode(self):
        '''
        Runs one episode of the environment, computes the GAE, and returns a tuple of
        (observations, actions, logps, advantages, returns) where:
            observations: [T, 25] a numpy array of observations
            actions: [T] a numpy array of actions
            logps: [T] a numpy array of log probabilities
            advantages: [T] a numpy array of advantages
            returns: [T] a numpy array of returns
        '''
        # sample a trajectory
        all_obs = []
        all_actions = []
        all_masks = []
        all_logps = []
        all_advantages = []
        all_returns = []

        player_1_states = []
        player_1_actions = []
        player_1_masks = []
        player_1_logps = []
        player_1_rewards = []
        
        player_2_states = []
        player_2_actions = []
        player_2_masks = []
        player_2_logps = []
        player_2_rewards = []

        env = Choko_Env()
        raw_obs, mask = env.reset()
        with torch.no_grad():
            while True:
                # (1) add the observation
                if env.player == 1:
                    obs = raw_obs
                    obs_torch = torch.from_numpy(raw_obs).float().unsqueeze(0)
                    player_1_states.append(obs)
                else:
                    obs = np.where(raw_obs == 0, 0, 3 - raw_obs)
                    obs_torch = torch.from_numpy(obs).float().unsqueeze(0)
                    player_2_states.append(obs)
                # (2) get the action and add the action

                torch_mask = torch.from_numpy(mask).float().unsqueeze(0)
                with torch.no_grad():
                    dist = self.agent(obs_torch, torch_mask)
                action = dist.sample()

                logp = dist.log_prob(action)
                logp_value = logp.item()

                if env.player == 1:
                    player_1_actions.append(action.item())
                    player_1_logps.append(logp_value)
                    player_1_masks.append(mask)
                else:
                    player_2_actions.append(action.item())
                    player_2_logps.append(logp_value)
                    player_2_masks.append(mask)

                state, reward, done, _ = env.step(action.item())
                raw_obs, mask = state

                # (3) add the reward (careful because player just switched after step)
                if env.player == 2:
                    player_1_rewards.append(reward)
                else:
                    player_2_rewards.append(reward)
                
                if done != "ongoing":
                    # the game is over, need to check if this player won because then the
                    # last reward for the opponent is -1.
                    if done == "won":
                        # the player who just played won
                        if env.player == 2:
                            # player 1 won, need to set the last reward for player 2 to -1
                            player_2_rewards[-1] = -2
                        else:
                            # player 2 won, need to set the last reward for player 1 to -1
                            player_1_rewards[-1] = -2
                    # add the last state to the trajectory
                    break
            
            player_1_states_np = np.stack(player_1_states, axis = 0)
            with torch.no_grad():
                player_1_values_tensor = self.critic(torch.as_tensor(player_1_states_np).float())
            player_1_values = player_1_values_tensor.tolist()
            player_1_values.append(0) # add the last value to the trajectory
            
            player_2_states_np = np.stack(player_2_states, axis = 0)
            with torch.no_grad():
                player_2_values_tensor = self.critic(torch.as_tensor(player_2_states_np).float())
            player_2_values = player_2_values_tensor.tolist()
            player_2_values.append(0)

            # now we have trajectories from each player's perspective. Need to add them to
            # experiences
            player_1_returns, player_1_advantages = utils.compute_gae_and_returns(
                player_1_rewards,
                player_1_values,
                self.gamma,
                self.lam
            )

            player_2_returns, player_2_advantages = utils.compute_gae_and_returns(
                    player_2_rewards,
                    player_2_values,
                    self.gamma,
                    self.lam
                )
            
            all_obs.extend(player_1_states)
            all_obs.extend(player_2_states)

            all_actions.extend(player_1_actions)
            all_actions.extend(player_2_actions)

            all_masks.extend(player_1_masks)
            all_masks.extend(player_2_masks)

            all_logps.extend(player_1_logps)
            all_logps.extend(player_2_logps)

            all_advantages.extend(player_1_advantages)
            all_advantages.extend(player_2_advantages)

            all_returns.extend(player_1_returns)
            all_returns.extend(player_2_returns)
            
            return all_obs, all_actions, all_masks, all_logps, all_advantages, all_returns
        
            



if __name__ == "__main__":
    env = Choko_Env()
    ppo_agent = PPOAgent(env.action_space.n, hidden_dim = 64)
    ppo_agent.switch_to_cpu()
    buffer = ReplayBuffer(0.99, 0.95, ppo_agent=ppo_agent, max_size = 5000)
    dataloader = buffer.make_dataloader(batch_size = 64, shuffle = True, num_workers = 0)
    for batch in dataloader:
        obs, actions, masks, logps, advantages, returns = batch
        print(obs.shape)
        print(actions.shape)
        print(masks.shape)
        print(logps.shape)
        print(advantages.shape)
        print(returns.shape)
        break


