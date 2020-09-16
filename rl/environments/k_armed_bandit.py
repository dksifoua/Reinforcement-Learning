import numpy as np
from rl.environments.base import Environment


class KArmedBanditEnv(Environment):

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.arms = np.zeros((self.n_actions,))

    def start(self):
        return None

    def step(self, action):
        reward = self.arms[action] + np.random.randn()
        return reward, None, False, {}

    def reset(self, seed=78):
        np.random.seed(seed)
        self.arms = np.random.randint(self.n_actions)
