import numpy as np
from envs.choko_env import Choko_Env

NUM_ACTIONS = (25) + (25 * 4) + (25 * 4 * 25) # 2625 possible actions
BOARD_DIM = 5

def compute_gae_and_returns(rewards, values, gamma, lam) -> tuple[list, list]:
    '''
    Computes and returns the GAE for one trajectory of length T of
    rewards and values.
    Args:
        rewards: [T] a numpy array of rewards
        values: [T + 1] a numpy array of values
        gamma: float, the discount factor
        lam: float, the lambda factor for GAE
    Returns:
        A tuple of (returns, advantages) where:
            returns: [T] a numpy array of returns
            advantages: [T] a numpy array of advantages
    '''
    T = len(rewards)
    advantages = [0.0] * T
    last_advantage = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantages[t] = delta + gamma * lam * last_advantage
        last_advantage = advantages[t]
    
    returns = [advantage + values[t] for t, advantage in enumerate(advantages)]
    return returns, advantages

def parse_inputs(inputs: list[str], env: Choko_Env) -> int:
    '''
    Parses the inputs from the player and returns the action.
    Args:
        inputs: a list of strings containing the action
    Returns:
        action: an integer representing the action
    '''
    try:
        if len(inputs) == 0:
            return -1
        action_type = inputs[0]
        if action_type == "place":
            if len(inputs) < 3:
                return -1
            row = int(inputs[1])
            col = int(inputs[2])
            if row < 0 or row >= BOARD_DIM or col < 0 or col >= BOARD_DIM:
                return -1
            if env.board[row][col] != 0:
                return -1
            action = row * BOARD_DIM + col
            return action
        elif action_type == "move":
            if len(inputs) < 4:
                return -1
            row = int(inputs[1])
            col = int(inputs[2])
            direction = inputs[3]
            if direction == "up":
                direction = 0
            elif direction == "right":
                direction = 1
            elif direction == "down":
                direction = 2
            elif direction == "left":
                direction = 3
            else:
                return -1
            if env.is_move_valid(row, col, direction, action_type):
                action = env.to_action(row, col, action_type, direction = direction)
                return action
            else:
                return -1
        elif action_type == "jump":
            if len(inputs) < 6:
                return -1
            row = int(inputs[1])
            col = int(inputs[2])
            direction = inputs[3]
            if direction == "up":
                direction = 0
            elif direction == "right":
                direction = 1
            elif direction == "down":
                direction = 2
            elif direction == "left":
                direction = 3
            else:
                return -1
            capture_row = int(inputs[4])
            capture_col = int(inputs[5])
            if env.is_move_valid(row, col, direction, action_type):
                action = env.to_action(
                    row, 
                    col, 
                    action_type, 
                    direction = direction, 
                    capture_row = capture_row, 
                    capture_col = capture_col
                    )
                return action
            else:
                return -1
        return -1
    except:
        return -1