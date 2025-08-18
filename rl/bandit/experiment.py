from typing import List, Tuple

import tqdm
import numpy as np

from rl.bandit import KArmedBanditEnvironment, KArmedBanditAgent


class KArmedBanditExperiment:
    __slots__ = (
        "environment",
        "n_runs",
        "n_steps",
    )

    def __init__(self, environment: KArmedBanditEnvironment, n_runs: int, n_steps: int) -> None:
        if n_runs <= 0 or n_steps <= 0:
            raise ValueError("n_runs and n_steps must be positive")

        self.environment = environment
        self.n_runs = n_runs
        self.n_steps = n_steps

    def start(self, agents: List[KArmedBanditAgent]) -> Tuple[np.ndarray, np.ndarray]:
        if not agents:
            raise ValueError("At least one agent must be provided")

        n_agents = len(agents)
        rewards = np.zeros((n_agents, self.n_runs, self.n_steps), dtype=np.float64)
        optimal_actions = np.zeros((n_agents, self.n_runs, self.n_steps), dtype=bool)

        for i, agent in enumerate(agents):
            for run in tqdm.tqdm(range(self.n_runs), desc=f"Agent-{i}"):
                self.environment.reset()
                agent.reset()

                for step in range(self.n_steps):
                    action = agent.act()
                    reward = self.environment.step(action=action)
                    agent.update(reward=reward)

                    rewards[i, run, step] = reward
                    optimal_actions[i, run, step] = (action == self.environment.best_action)

        return rewards.mean(axis=1), optimal_actions.mean(axis=1) * 100
