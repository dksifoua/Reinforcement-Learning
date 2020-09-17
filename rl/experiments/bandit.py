import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from rl.agents.bandit import BanditAgent
from rl.environments.bandit import BanditEnvironment


class BanditExperiment:
    """

    Args:
        environment (BanditEnvironment): the bandit environment.
        n_runs (int): the number of runs.
        n_time_steps (int): the number of time steps.
    """

    def __init__(self, environment: BanditEnvironment, n_runs: int, n_time_steps: int):
        self.environment = environment
        self.n_runs = n_runs
        self.n_time_steps = n_time_steps

    def run(self, agents: List[BanditAgent]) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.zeros((len(agents), self.n_runs, self.n_time_steps))
        best_action_counts = np.zeros(rewards.shape)
        for i, agent in enumerate(agents):
            for run in range(self.n_runs):
                self.environment.reset()
                agent.reset()
                for time_step in range(self.n_time_steps):
                    action = agent.act()
                    reward = self.environment.step(action)
                    agent.update_q_values(reward)
                    if action == self.environment.best_action:
                        best_action_counts[i, run, time_step] += 1
                    rewards[i, run, time_step] += reward
        return rewards.mean(axis=1), best_action_counts.mean(axis=1)

    def get_reward_distribution(self):
        plt.figure(figsize=(15, 5))
        plt.violinplot(
            dataset=np.random.randn(200, self.environment.n_actions) + np.random.randn(self.environment.n_actions))
        plt.xlabel('Action')
        plt.ylabel('Reward distribution')
        plt.show()
