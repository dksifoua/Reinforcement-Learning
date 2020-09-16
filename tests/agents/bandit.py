import unittest
import numpy as np
from rl.agents.bandit import EpsilonGreedyAgent, EpsilonGreedyAgentWithConstantStepSize, GreedyAgent
from rl.agents.utils import argmax


class TestBanditAgent(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_argmax(self):
        self.assertEqual(argmax(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])), 8)
        self.assertIn(argmax(np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 0])), [0, 3, 5, 8])

    def test_greedy_agent(self):
        agent = GreedyAgent()
        agent.q_values = np.array([0, 0, 1.0, 0, 0])
        agent.arm_count = np.array([0, 1, 0, 0, 0])
        agent.last_action = 1
        action = agent.step(reward=1, state=None)
        self.assertEqual(action, 2)
        self.assertTrue(np.array_equal(agent.q_values, np.array([0, 0.5, 1.0, 0, 0])))

    def test_epsilon_greedy_agent(self):
        agent = EpsilonGreedyAgent()
        agent.q_values = np.array([0, 0, 1.0, 0, 0])
        agent.arm_count = np.array([0, 1, 0, 0, 0])
        agent.num_actions = 5
        agent.last_action = 1
        agent.epsilon = 0.5
        _ = agent.step(reward=1, state=None)
        self.assertTrue(np.array_equal(agent.q_values, np.array([0, 0.5, 1.0, 0, 0])))

    def test_epsilon_greedy_agent_with_constant_step_size(self):
        for step_size in [0.01, 0.1, 0.5, 1.0]:
            agent = EpsilonGreedyAgentWithConstantStepSize()
            agent.q_values = np.array([0, 0, 1.0, 0, 0])
            agent.num_actions = 5
            agent.last_action = 1
            agent.epsilon = 0.0
            agent.step_size = step_size
            _ = agent.step(reward=1, state=None)
            self.assertTrue(np.array_equal(agent.q_values, np.array([0, step_size, 1.0, 0, 0])))

    def test_softmax_agent(self):
        # TODO
        pass

    def test_softmax_agent_with_constant_step_size(self):
        # TODO
        pass

    def test_upper_confidence_bound_agent(self):
        # TODO
        pass
