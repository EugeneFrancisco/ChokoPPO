from envs.choko_env import Choko_Env
from agents.ppo_agent import PPOAgent
from torch.distributions import Categorical
from infrastructure import utils
import torch
import random

NUM_ACTIONS = (25) + (25 * 4) + (25 * 4 * 25) # 2625 possible actions
HIDDEN_DIM = 64

def run_game():
    print("Starting game...")
    print("When prompted for an action, input the action in the following format:\n")
    print("'action_name, first_row first_col direction capture_row capture_col' (space separated)\n")
    print("For example, 'move 0 0 right' means moving the piece at (0, 0) to the right.\n")
    print("The capture_row and capture_col are for capturing a piece.\n")
    print("Invalid actions will lead to reprompting.\n")
    env = Choko_Env()
    agent = PPOAgent(num_actions = NUM_ACTIONS, hidden_dim = HIDDEN_DIM)
    # agent.load_state_dict(torch.load("models/ppo_agent.pth"))
    agent.eval()
    agent.switch_to_cpu()
    raw_obs, mask = env.reset()

    # TODO, fix this turn sequence. tbh idrk what is wrong
    if random.random() < 0.5:
        env.player = 2
    
    while True:
        if env.player == 1:
            # agent's turn
            obs = torch.tensor(raw_obs, dtype=torch.float32).unsqueeze(0)
            torch_mask = torch.from_numpy(mask).unsqueeze(0)
            dist = agent.act(obs, torch_mask)
            action = dist.sample()
            state, _, done, _ = env.step(action)
            raw_obs, mask = state
            if done:
                print("Agent wins!")
                break
        else:
            # player's turn
            print("Your turn! The board looks like this (You are X):\n")
            env.render() # TODO
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
            if done:
                print("Player wins!")
                break

    

if __name__ == "__main__":
    run_game()