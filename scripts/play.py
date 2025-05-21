from envs.choko_env import Choko_Env
from agents.ppo_agent import PPOAgent
from torch.distributions import Categorical
from infrastructure import utils
import torch
import numpy as np
import random

NUM_ACTIONS = (25) + (25 * 4) + (25 * 4 * 25) # 2625 possible actions
HIDDEN_DIM = 64

def user_v_agent(ppo_agent):
    print("Starting game...")
    print("When prompted for an action, input the action in the following format:\n")
    print("'action_name, first_row first_col direction capture_row capture_col' (space separated)\n")
    print("For example, 'move 0 0 right' means moving the piece at (0, 0) to the right.\n")
    print("The capture_row and capture_col are for capturing a piece.\n")
    print("Invalid actions will lead to reprompting.\n")
    env = Choko_Env()
    agent = ppo_agent
    raw_obs, mask = env.reset()
    
    while True:
        # agent's turn
        obs = torch.tensor(raw_obs, dtype=torch.float32).unsqueeze(0)
        torch_mask = torch.from_numpy(mask).unsqueeze(0)
        dist = agent.act(obs, torch_mask)
        action = dist.sample()
        info_map = env.to_info(action.item())
        print("\nAgent's move:")
        print(info_map["move_type"])
        print("\n")
        state, _, done, _ = env.step(action.item())
        raw_obs, mask = state
        if done != "ongoing":
            print("Agent wins!")
            break


        # player's turn
        print("Your turn! The board looks like this (You are X):\n")
        env.render() 
        print("\n")
        action_input = input("Input your action in the format printed: ")
        inputs = action_input.split(" ")
        action = utils.parse_inputs(inputs, env)
        while action == -1:
            print("Invalid action! Try again.")
            action_input = input("Your turn! Input your action in the format printed: ")
            inputs = action_input.split(" ")
            action = utils.parse_inputs(inputs, env)
        state, _, done, _ = env.step(action)
        raw_obs, mask = state
        if done != "ongoing":
            print("Player wins!")
            break

def agent_v_agent(ppo_agent_1, ppo_agent_2):
    '''
    Simulates a game between two agents and returns the winner (1 for player 1, 2 for player 2, -1 for draw)
    '''
    env = Choko_Env()
    raw_obs, mask = env.reset()
    while True:
        # agent 1's turn
        obs = torch.tensor(raw_obs, dtype=torch.float32).unsqueeze(0)
        torch_mask = torch.from_numpy(mask).unsqueeze(0)
        dist = ppo_agent_1.act(obs, torch_mask)
        action = dist.sample()
        state, _, done, _ = env.step(action.item())
        raw_obs, mask = state
        if done != "ongoing":
            if done == "draw":
                return -1
            return 1

        # agent 2's turn
        obs = np.where(raw_obs == 0, 0, 3 - raw_obs)
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        torch_mask = torch.from_numpy(mask).unsqueeze(0)
        dist = ppo_agent_2.act(obs, torch_mask)
        action = dist.sample()
        info_map = env.to_info(action.item())
        state, _, done, _ = env.step(action.item())
        raw_obs, mask = state
        if done != "ongoing":
            if done == "draw":
                return -1
            return 2

    

if __name__ == "__main__":
    agent_new = PPOAgent(num_actions = NUM_ACTIONS, hidden_dim = HIDDEN_DIM)
    agent_new.switch_to_cpu()
    checkpoint = torch.load(
        "checkpoints/ppo_agent_1990.pth",
        map_location=agent_new.device,         # ensures weights land on the right device
        weights_only=True                  # suppresses the FutureWarning by only loading tensors
    )
    agent_new.load_state_dict(checkpoint["model_state_dict"])
    agent_new.eval()
    
    agent_old = PPOAgent(num_actions = NUM_ACTIONS, hidden_dim = HIDDEN_DIM)
    agent_old.switch_to_cpu()
    checkpoint = torch.load(
        "checkpoints/ppo_agent_50.pth",
        map_location=agent_old.device,         # ensures weights land on the right device
        weights_only=True                  # suppresses the FutureWarning by only loading tensors
    )
    agent_old.load_state_dict(checkpoint["model_state_dict"])
    agent_old.eval()

    first_win_count = 0
    second_win_count = 0
    draw_count = 0
    for i in range(100):
        outcome = agent_v_agent(agent_old, agent_new)
        if outcome == 1:
            first_win_count += 1
        elif outcome == 2:
            second_win_count += 1
        else:
            draw_count += 1
    print(f"Agent 1 wins: {first_win_count}")
    print(f"Agent 2 wins: {second_win_count}")
    print(f"Draws: {draw_count}")

    #user_v_agent(agent_new)



