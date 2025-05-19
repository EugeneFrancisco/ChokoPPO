import numpy as np


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