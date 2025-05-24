from envs.choko_env import Choko_Env
from agents.random_agent import RandomAgent
from agents.ppo_agent import PPOAgent
from torch.utils.data import Dataset, DataLoader
from . import utils
import numpy as np
import torch
import torch.nn as nn
import tqdm
import config


class RLDataset(Dataset):
    # Too lazy to change names but this is the RLDataset for the PPO agent, NOT the Q learner.
    def __init__(self, obs, actions, masks, logps, advantages, returns):
        self.obs = torch.from_numpy(obs).float()
        self.actions = torch.from_numpy(actions)
        masks = np.array(masks, dtype=np.bool_)
        self.masks = torch.from_numpy(masks)
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
    # Too lazy to change names but this is the ReplayBuffer for the PPO agent, NOT the Q learner.
    def __init__(self, gamma, lam, ppo_agent, max_size = config.MAX_BUFFER_SIZE):
        self.gamma = gamma
        self.lam = lam
        self.max_size = max_size
        self.ppo_agent = ppo_agent
        self.agent = self.ppo_agent.act
        self.critic = self.ppo_agent.critic
    
    def refresh(self, ppo_agent):
        '''
        Refreshes the agent and the critic in the buffer.
        This is useful when the agent is updated and we want to use the new agent for collecting rollouts.
        '''
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
            persistent_workers=True,
            pin_memory=False,
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

            rollout_obs, rollout_actions, rollout_masks, rollout_logps, rollout_advantages, rollout_returns = self.run_one_episode()

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

            # Note, removed torch.no_grad() here because I wrapped the whole function in it
            player_1_values_tensor = self.critic(torch.as_tensor(player_1_states_np).float())
            player_1_values = player_1_values_tensor.tolist()
            player_1_values.append(0) # add the last value to the trajectory
            
            player_2_states_np = np.stack(player_2_states, axis = 0)

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
        
            
class ReplayBufferQLearning:
    def __init__(self, gamma, critic, max_size = config.MAX_BUFFER_SIZE):
        self.gamma = gamma
        self.max_size = max_size
        self.critic = critic
        self.td_gap = 5
    
    def make_dataloader(self, batch_size = 64, shuffle = True, num_workers = 0) -> DataLoader:
        # TODO
        pass

    def make_dataset(self):
        '''
        Returns a tuple of (dataset_obs, actions, masks, bootstrap_targets, next_states) where:
            dataset_obs: [max_size, 25] a numpy array of observations
            actions: [max_size] a numpy array of actions
            masks: [max_size] a numpy array of masks for which actions are valid at the state
            bootstrap_targets: [max_size] a numpy array of bootstrap targets
            next_states: [max_size, 25] a numpy array of next states
        '''
        dataset_obs = []
        actions = []
        masks = []
        bootstrap_targets = []
        next_states = []

        while len(dataset_obs) < self.max_size:
            # TODO
            break
        pass

    def run_one_episode(self):
        '''
        Runs one episode (game) of the environment, computes the bootstrap targets, and returns a tuple of
        (observations, actions, masks, bootstrap_targets, next_states) where:
            observations: [T, 25] a list of numpy observations
            actions: [T] a list of actions
            masks: [T] a list of masks for which actions are valid at the state
            bootstrap_targets: [T] a list of bootstrap targets
        '''
        all_obs = []
        all_actions = []
        all_masks = []
        all_bootstrap_targets = []
        
        player_1_states = []
        player_1_actions = []
        player_1_masks = []
        player_1_bootstrap_targets = []

        player_2_states = []
        player_2_actions = []
        player_2_masks = []
        player_2_bootstrap_targets = []

        player_1_buffer = ExperienceBuffer(self.gamma, self.td_gap, self.critic)
        player_2_buffer = ExperienceBuffer(self.gamma, self.td_gap, self.critic)

        env = Choko_Env()
        raw_obs, mask = env.reset()
        
        with torch.no_grad():
            while True:
                if env.player == 1:
                    obs = raw_obs
                    obs_torch = torch.from_numpy(raw_obs).float().unsqueeze(0)
                else:
                    obs = np.where(raw_obs == 0, 0, 3 - raw_obs)
                    obs_torch = torch.from_numpy(obs).float().unsqueeze(0)
                
                torch_mask = torch.from_numpy(mask).float().unsqueeze(0)
                action = self.critic.sample_action(obs_torch, torch_mask) 

                state_tup, reward, done, _ = env.step(action.item())


                if done != "ongoing":
                    if done == "won":
                        winner_reward = 2 + reward
                        loser_reward = -2
                    else:
                        # draw
                        winner_reward = 0
                        loser_reward = 0

                    if env.player == 1:
                        target_vals = player_1_buffer.add((obs, obs_torch, action, mask, winner_reward))
                        assert target_vals is not None
                        player_1_states.append(target_vals[0])
                        player_1_actions.append(target_vals[1])
                        player_1_masks.append(target_vals[2])
                        player_1_bootstrap_targets.append(target_vals[3])

                        flused_obs_1, flushed_actions_1, flushed_masks_1, flushed_bootstrap_targets_1 = player_1_buffer.flush_buffer()
                        player_1_states.extend(flused_obs_1)
                        player_1_actions.extend(flushed_actions_1)
                        player_1_masks.extend(flushed_masks_1)
                        player_1_bootstrap_targets.extend(flushed_bootstrap_targets_1)

                        # last experience in loser's buffer has edited reward because they lost
                        target_vals = player_2_buffer.buffer[len(player_2_buffer.buffer) - 1][4] = loser_reward
                        flushed_obs_2, flushed_actions_2, flushed_masks_2, flushed_bootstrap_targets_2 = player_2_buffer.flush_buffer()
                        player_2_states.extend(flushed_obs_2)
                        player_2_actions.extend(flushed_actions_2)
                        player_2_masks.extend(flushed_masks_2)
                        player_2_bootstrap_targets.extend(flushed_bootstrap_targets_2)
                    
                    if env.player == 2:
                        target_vals = player_2_buffer.add((obs, obs_torch, action, mask, 2 + reward))
                        assert target_vals is not None
                        player_2_states.append(target_vals[0])
                        player_2_actions.append(target_vals[1])
                        player_2_masks.append(target_vals[2])
                        player_2_bootstrap_targets.append(target_vals[3])

                        flushed_obs_2, flushed_actions_2, flushed_masks_2, flushed_bootstrap_targets_2 = player_2_buffer.flush_buffer()
                        player_2_states.extend(flushed_obs_2)
                        player_2_actions.extend(flushed_actions_2)
                        player_2_masks.extend(flushed_masks_2)
                        player_2_bootstrap_targets.extend(flushed_bootstrap_targets_2)

                        # last experience in loser's buffer has edited reward because they lost
                        target_vals = player_1_buffer.buffer[len(player_1_buffer.buffer) - 1][4] = loser_reward
                        flushed_obs_1, flushed_actions_1, flushed_masks_1, flushed_bootstrap_targets_1 = player_1_buffer.flush_buffer()
                        player_1_states.extend(flushed_obs_1)
                        player_1_actions.extend(flushed_actions_1)
                        player_1_masks.extend(flushed_masks_1)
                        player_1_bootstrap_targets.extend(flushed_bootstrap_targets_1)
                    
                    break

                
                if env.player == 1:
                    target_vals = player_1_buffer.add((obs, obs_torch, action, mask, reward))
                    if target_vals is not None:
                        player_1_states.append(target_vals[0])
                        player_1_actions.append(target_vals[1])
                        player_1_masks.append(target_vals[2])
                        player_1_bootstrap_targets.append(target_vals[3])
                else:
                    target_vals = player_2_buffer.add((obs, obs_torch, action, mask, reward))
                    if target_vals is not None:
                        player_2_states.append(target_vals[0])
                        player_2_actions.append(target_vals[1])
                        player_2_masks.append(target_vals[2])
                        player_2_bootstrap_targets.append(target_vals[3])

                raw_obs, mask = state_tup
        
        all_obs.extend(player_1_states)
        all_actions.extend(player_1_actions)
        all_masks.extend(player_1_masks)
        all_bootstrap_targets.extend(player_1_bootstrap_targets)

        all_obs.extend(player_2_states)
        all_actions.extend(player_2_actions)
        all_masks.extend(player_2_masks)
        all_bootstrap_targets.extend(player_2_bootstrap_targets)

        return all_obs, all_actions, all_masks, all_bootstrap_targets
                


class ExperienceBuffer:
    def __init__(self, gamma, size, critic):
        self.size = size
        self.buffer = [] # each element in the buffer is a tuple of (state, action, reward)
        self.critic = critic
        self.gamma = gamma
        
    def add(self, experience_tuple):
        '''
        Given an experience_tuple of (state (numpy), state (torch), action, mask, reward), adds it to the buffer.
        If adding the experience makes the buffer full or exceed size, calculates the bootstrapped target
        and replaces the oldest experience in the buffer with the new experience. Also returns the bootstrapped target.
        Args:
            experience_tuple: a tuple of (state (numpy), state (torch), action, reward)
        Returns:
            a tuple (state (numpy), action, bootstrapped_target): 
                state: the first state in the buffer
                action: the first action in the buffer
                the bootstrapped target for the experience, or none if the buffer is not full
            or None if the buffer is not full.
        '''
        # TODO, test this
        state, state_torch, action, mask, reward = experience_tuple
        self.buffer.append(experience_tuple)
        if len(self.buffer) > self.size:
            # calculate the bootstrapped target
            bootstrapped_target = 0
            for i in range(len(self.buffer) - 1):
                _, _, _, _, reward = self.buffer[i]
                bootstrapped_target += reward * (self.gamma ** i)
            
            last_experience = self.buffer[-1]
            _, state_torch, action, mask, _ = last_experience
            # TODO CRITIC POLICY
            critic_values = self.critic(state_torch)
            bootstrapped_target += self.gamma ** len(self.buffer) * self.critic.get_value(state_torch, action, mask)
            first_state, _, first_action, mask, _ = self.buffer.pop(0)
            return (first_state, first_action, mask, bootstrapped_target)
        
        return None
    
    def flush_buffer(self) -> np.ndarray[int]:
        '''
        Empties the remaining experiences in the buffer and calculates the bootstrapped targets for each experience.
        Ideally this should be called at the end of the episode when the last experience in the buffer
        is the last non-terminal experience. #TODO, is this right?
        Returns:
            (obs, actions, masks, bootstrapped_targets) where:
                obs: a numpy array of observations for each experience in the buffer
                actions: a numpy array of actions for each experience in the buffer
                masks: a numpy array of masks for each experience in the buffer
                bootstraped_targets: a numpy array of bootstrapped targets for each experience in the buffer, where
                the targets are calculated as the weighted sum of rewards starting from each experience.

        '''

        obs = []
        actions = []
        masks = []
        bootstrapped_targets = []



        for i in range(len(self.buffer)):
            target = sum(self.buffer[j][2]*(self.gamma ** j) for j in range(i, len(self.buffer)))
            obs.append(self.buffer[i][0])
            actions.append(self.buffer[i][2])
            masks.append(self.buffer[i][3])
            bootstrapped_targets.append(target)
        
        self.buffer = []

        return obs, actions, masks, bootstrapped_targets

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


