import numpy as np
from rl.agents.base import Agent
from rl.agents.utils import argmax


class BaseAgent(Agent):

    def __init__(self):
        self.n_actions = None
        self.step_size = None
        self.epsilon = None
        self.C = None
        self.time_step = None
        self.q_values = None
        self.arm_count = None
        self.last_action = None

    def reset(self, **kwargs):
        self.n_actions = kwargs.get('n_actions', 10)
        self.step_size = kwargs.get('step_size', 0.1)
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.C = kwargs.get('C', 2)
        self.time_step = 0
        self.q_values = np.ones((self.n_actions,)) * kwargs.get('initial_values', 0.0)
        self.arm_count = np.zeros((self.n_actions,))
        self.last_action = 0

    def start(self, state, **kwargs):
        self.last_action = np.random.choice(self.n_actions)
        return self.last_action

    def step(self, reward, state, **kwargs):
        self.last_action = np.random.choice(self.n_actions)
        return self.last_action

    def end(self, reward, state, **kwargs):
        pass


class GreedyAgent(BaseAgent):
    """
    Agent that implements greedy action selection.
    """

    def step(self, reward, state, **kwargs):
        action = argmax(self.q_values)
        self.arm_count[self.last_action] += 1
        self.q_values[self.last_action] += (1 / self.arm_count[self.last_action]) * (
                reward - self.q_values[self.last_action])
        self.last_action = action
        return action


class EpsilonGreedyAgent(BaseAgent):
    """
    Agent that implements $\varepsilon$-greedy action selection.
    """

    def step(self, reward, state, **kwargs):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.q_values.size)
        else:
            action = argmax(self.q_values)
        self.arm_count[self.last_action] += 1
        self.q_values[self.last_action] += (1 / self.arm_count[self.last_action]) * (
                reward - self.q_values[self.last_action])
        self.last_action = action
        return action


class EpsilonGreedyAgentWithConstantStepSize(BaseAgent):
    """
    Agent that implements $\varepsilon$-greedy action selection with constant step-size.
    """

    def step(self, reward, state, **kwargs):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.q_values.size)
        else:
            action = argmax(self.q_values)
        self.q_values[self.last_action] += self.step_size * (reward - self.q_values[self.last_action])
        self.last_action = action
        return action


class SoftmaxAgent(BaseAgent):
    """
    Agent that implements softmax action selection.
    """
    # TODO
    #   Implements agent step using softmax
    pass


class SoftmaxAgentWithConstantStepSize(BaseAgent):
    """
    Agent that implements softmax action selection with constant step-size.
    """
    # TODO
    #   Implements agent step using softmax action selection
    pass


class UpperConfidenceBoundAgent(BaseAgent):
    # TODO
    #   Implements agent step using Upper Confidence Bound (UCB) action selection
    pass
