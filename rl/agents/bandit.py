import numpy as np
from rl.agents.utils import argmax


class BanditAgent:
    """
    Args:
        n_actions (int): the number of action the agent can take.
        initial_value (float): the initial action value for all actions.
        epsilon (float): the probability for exploration in epsilon-greedy algorithm.
        step_size (float): the constant step-size for updating action values estimates.
        sample_average (bool): if True, use sample averages to update action values estimates else, use const step-size
        C (float): this param controls the exploration in case of UCB algorithm.
        time_step (int): the time step.
        q_values (np.ndarray[float]): the action values estimates.
        action_count (np.ndarray[int]): the number of time each action is taken by the agent.
        last_action (int): the last action taken by the agent.
        average_reward (float): the average reward.
    """

    def __init__(self, n_actions, initial_value=0., epsilon=0., step_size=None, sample_average=False, C=None,
                 time_step=0, q_values=None, action_count=None, last_action=None, average_reward=None):
        self.n_actions = n_actions
        self.initial_value = initial_value
        self.epsilon = epsilon
        self.step_size = step_size
        self.sample_average = sample_average
        self.C = C
        self.time_step = time_step
        self.q_values = q_values
        self.action_count = action_count
        self.last_action = last_action
        self.average_reward = average_reward

    def reset(self) -> None:
        self.q_values = np.zeros((self.n_actions,)) + self.initial_value
        self.action_count = np.zeros((self.n_actions,))
        self.time_step = 0

    def choose_action(self) -> int:
        if np.random.randn() < self.epsilon:
            return np.random.choice(np.arange(self.n_actions))
        if self.C is not None:
            return argmax(self.q_values + self.C * np.sqrt(np.log(self.time_step + 1) / (self.action_count + 1e-5)))
        return argmax(self.q_values)

    def start(self) -> int:
        self.last_action = self.choose_action()
        return self.last_action

    def step(self, reward):
        self.time_step += 1
        self.average_reward += (reward - self.average_reward) / self.time_step
        if self.sample_average:
            N = self.action_count[self.last_action]
            self.q_values[self.last_action] += (reward - self.q_values[self.last_action]) / N
        else:
            self.q_values[self.last_action] += self.step_size * (reward - self.q_values[self.last_action])
        self.last_action = self.choose_action()
        self.action_count[self.last_action] += 1
        return self.last_action
