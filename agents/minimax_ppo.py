import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from envs.choko_env_old import Choko_Env #TODO change to new env
from agents.ppo_agent import PPOAgent

from infrastructure.utils import _flip_obs

class MinimaxPPOTreeSearch:
    def __init__(self, agent: PPOAgent, decay: bool, depth = 5, k = 5):
        self.agent = agent
        self.depth = depth
        self.k = k
        self.decay = decay
    
    def select_action(self, env: Choko_Env) -> int:
        best_value, best_action = self.select_value_and_action(env, self.depth, env.player, self.k)
        if best_action == -1:
            raise ValueError("No valid action found!")
        return best_action
    
    def select_value_and_action(self, env: Choko_Env, depth: int, player: int, k: int) -> int:
        root_obs, root_mask = env.fetch_obs_action_mask()
        # base case, if depth is 0 or game is over, can use the env method to evaluate termination
        # base case, return the reward if the game is over because we have won/lost,
        # return the value if the game is not over yet.

        # recursive case, query the policy for the actions and apply the observation mask
        # pick the top k actions based on the policy
        # for each action, make a copy of the environment, apply the action,
        # and recurse.
        if env.player == 2:
            root_obs = _flip_obs(root_obs, False)
        root_obs_torch = torch.from_numpy(root_obs).float().unsqueeze(0)
        root_mask_torch = torch.from_numpy(root_mask).float().unsqueeze(0)
        if depth == 0 or env.evaluate_termination() != "ongoing":
            if env.evaluate_termination() != "ongoing":
                if env.evaluate_termination() == "draw":
                    return 0, -1
                if env.player == player:
                    # the player who just won was not us, so we return a negative reward
                    return -2, -1
                else:
                    # we just won, but the positive reward is handled in the recursive logic.
                    return 2, -1
            else:
                return self.agent.critic(root_obs_torch).item(), -1
        
        # now the recursive case

        dist = self.agent.act(root_obs_torch, root_mask_torch)
        probs = dist.probs.squeeze(0)
        
        best_value = -1e10
        best_action = -1

        topk_probs, topk_idxs = probs.topk(k, dim=-1)
        count = 0
        for action, prob in zip(topk_idxs, topk_probs):
            count += 1
            if root_mask[action.item()] == 0:
                continue # skip invalid actions
            env_copy = copy.deepcopy(env)
            _, reward, _, _ = env_copy.step(action.item())
            
            if self.decay:
                if k > 4:
                    new_k = int(k * 1/2)
                else:
                    new_k = k
            else:
                new_k = k

            best_child_value, best_child_action = self.select_value_and_action(
                env_copy, depth - 1, player, new_k
            )
            if env_copy.player == player:
                # this action just taken was not our turn, so we negate the reward
                reward = -reward
            this_value = (reward - best_child_value)
            if this_value > best_value:
                best_value = this_value
                best_action = action.item()
        
        if count > 0 and best_action == -1:
            best_action = topk_idxs[0].item()
        return best_value, best_action



        