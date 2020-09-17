import numpy as np


class BanditEnvironment:
    """
    Args:
        n_actions (int): the number of actions (arms).
        reward (float): the initial reward value.
        rewards (np.ndarray[float], optional): the reward value of each action (follows N(reward, 1)).
        best_action (float, optional): the action that gives the best reward in the environment.
    """

    def __init__(self, n_actions, reward=0., rewards=None, best_action=None):
        self.n_actions = n_actions
        self.reward = reward
        self.rewards = rewards
        self.best_action = best_action

    def reset(self):
        self.rewards = np.random.randn(self.n_actions) + self.reward
        self.best_action = self.rewards.argmax()

    def step(self, action) -> float:
        reward = self.rewards[action] + np.random.randn()  # Generate a reward that follows N(real reward, 1)
        return reward
