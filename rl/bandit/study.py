from typing import Tuple, Dict

import numpy as np

from rl.bandit.agent import BanditAgentArguments, BanditGradientAgentArguments
from rl.bandit.environment import BanditEnvironmentArguments
from rl.bandit.experiment import BanditExperiment


class BanditStudy:
    __slots__ = ("n_actions",)

    def __init__(self, n_actions: int) -> None:
        if n_actions < 2:
            raise ValueError("n_actions must greater than 2.")

        self.n_actions = n_actions

    def start(self, n_steps: int, n_runs: int, n_samples: int
              ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if n_runs < 1 or n_steps < 1 or n_samples < 1:
            raise ValueError("n_runs, n_steps, and n_samples must be strictly positive.")

        experiment = BanditExperiment(environment_args=BanditEnvironmentArguments(n_actions=self.n_actions),
                                      n_steps=n_steps, n_runs=n_runs)

        epsilons = np.linspace(start=1 / 128, stop=1 / 4, num=n_samples)
        step_sizes = np.linspace(start=1 / 32, stop=5 / 2, num=n_samples)
        ucbs = np.linspace(start=1 / 16, stop=4, num=n_samples)
        q0s = np.linspace(start=1 / 4, stop=4, num=n_samples)

        rewards, _ = experiment.start(
            agent_args_list=[
                                BanditAgentArguments(n_actions=self.n_actions, epsilon=epsilon)
                                for epsilon in epsilons
                            ] + [
                                BanditGradientAgentArguments(n_actions=self.n_actions, step_size=step_size,
                                                             baseline=True)
                                for step_size in step_sizes
                            ] + [
                                BanditAgentArguments(n_actions=self.n_actions, epsilon=0.0, c=c)
                                for c in ucbs
                            ] + [
                                BanditAgentArguments(n_actions=self.n_actions, epsilon=0.0, step_size=0.1,
                                                     initial_action_value=q0)
                                for q0 in q0s
                            ]
        )
        return {
            "epsilon": rewards[:n_samples, :].mean(axis=1),
            "step_size": rewards[n_samples:n_samples * 2, :].mean(axis=1),
            "ucb": rewards[n_samples * 2:n_samples * 3, :].mean(axis=1),
            "q0": rewards[n_samples * 3:n_samples * 4, :].mean(axis=1),
        }, epsilons, step_sizes, ucbs, q0s
