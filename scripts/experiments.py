from envs.choko_env_old import Choko_Env as Choko_Env_Old
from envs.choko_env import Choko_Env
from agents.ppo_agent import PPOAgent, PPOAgentOld
from agents.q_agent import QAgent
from agents.minimax_agent import MinimaxAgent
from torch.distributions import Categorical
from agents.minimax_ppo import MinimaxPPOTreeSearch
from infrastructure import utils
import torch
import numpy as np
import random
import config
from scripts import play

NUM_ACTIONS = (25) + (25 * 4) + (25 * 4 * 25) # 2625 possible actions
HIDDEN_DIM = 64

def user_v_mcts(mcts):
    print("Starting game...")
    print("When prompted for an action, input the action in the following format:\n")
    print("'action_name, first_row first_col direction capture_row capture_col' (space separated)\n")
    print("For example, 'move 0 0 right' means moving the piece at (0, 0) to the right.\n")
    print("The capture_row and capture_col are for capturing a piece.\n")
    print("Invalid actions will lead to reprompting.\n")
    env = Choko_Env()
    raw_obs, mask = env.reset()

    turn = None
    
    while True:
        # agent's turn
        if turn is None or turn != "user":
            action = mcts.select_action(env)
            if action is None:
                print("MCTS agent could not find a valid action!")
                break
            info_map = env.to_info(action)
            print("\nAgent's move:")
            print(info_map["move_type"])
            print("\n")
            _, _, done, _ = env.step(action)
            print("\n")
            if done != "ongoing":
                print("Agent wins!")
                break


        # player's turn
        print("Your turn! The board looks like this (You are O).")
        print(f"You have {env.pieces_left[env.player]} pieces left.\n")
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
        print("\n")
        env.render()
        print("\n")
        raw_obs, mask = state
        if done != "ongoing":
            print("Player wins!")
            break
        turn = None

def copy_game_state(env: Choko_Env, info: dict):
    '''
    Given a dictionary of game info, copies the dictionary info into the environment.
    info must contain:
        - board: np.ndarray of shape (5, 5)
        - player: the player whose turn it is
        - pieces_left: dictionary {1: # pieces left to place for player 1,
                                    2: # pieces left to place for player 2}
        - pieces_captured: dictionary {1: # of player 1's pieces captured,
                                       2: # of player 2's pieces captured}
        -  drop_initiative: 0, 1, or 2
        -  num_moves: number of moves played so far
    '''
    env.board = info["board"].copy()
    env.player = info["player"]
    env.pieces_left = info["pieces_left"].copy()
    env.pieces_captured = info["pieces_captured"].copy()
    env.drop_initiative = info["drop_initiative"]
    env.num_moves = info["num_moves"]


if __name__ == "__main__":
    agent = PPOAgent(num_actions = config.NUM_ACTIONS,  hidden_dim = HIDDEN_DIM)
    agent.switch_to_cpu()
    checkpoint = torch.load(
        "checkpoints/ppo/run_13/ppo_agent_4250.pth",
        map_location=agent.device,         # ensures weights land on the right device
        weights_only=True                  # suppresses the FutureWarning by only loading tensors
    )
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    # env = Choko_Env()
    # board = np.array([
    #     [1, 2, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 1, 0, 1, 0],
    #     [0, 0, 0, 2, 0],
    #     [1, 0, 0, 0, 0]
    # ])
    # drop_initiative = 2
    # pieces_left = {1: 0, 2: 0}
    # pieces_captured = {1: 0, 2: 2}
    # num_moves = 0
    # player = 2

    # info = {
    #     "board": board,
    #     "player": player,
    #     "pieces_left": pieces_left,
    #     "pieces_captured": pieces_captured,
    #     "drop_initiative": drop_initiative,
    #     "num_moves": num_moves
    # }

    # copy_game_state(env, info)
    # obs, mask = env.fetch_obs_action_mask()
    
    # obs = utils._flip_obs(obs, False)
    # torch_obs = torch.from_numpy(obs).float().unsqueeze(0)
    # torch_mask = torch.from_numpy(mask).float().unsqueeze(0)
    # dist = agent.act(torch_obs, torch_mask)
    # probs = dist.probs.squeeze(0)

    # print("PPO V2 top probs:")
    # topk_probs, topk_idxs = probs.topk(4, dim =-1)
    # for prob, idx in zip(topk_probs, topk_idxs):
    #     action = idx.item()
    #     info_map = env.to_info(action)
    #     print(f"Action: {info_map['move_type']}, Probability: {prob.item():.4f}")
    #     print(f"Action details: {info_map}\n")

    minimax_ppo = MinimaxPPOTreeSearch(agent, depth=5, k=10, decay=True)
    #minimax_agent = MinimaxAgent(max_depth=3)

    # action = minimax_ppo.select_action(env)
    # info_map = env.to_info(action)
    # print("\nMinimax Agent's move:")
    # print(f"Action: {info_map['move_type']}")
    # print(f"Action details: {info_map}\n")

    # action = minimax_agent.choose_action(env)
    # info_map = env.to_info(action)
    # print("\nMinimax Agent's move (legacy):")
    # print(f"Action: {info_map['move_type']}")
    # print(f"Action details: {info_map}\n")

    user_v_mcts(minimax_ppo)

