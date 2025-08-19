from typing import List, Optional

import numpy as np

from rl.bandit.utils import guard_agent_act, guard_agent_update, guard_agent_init


class BanditAgent:
    __slots__ = (
        "n_actions",
        "epsilon",
        "rng",
        "step",
        "action",
        "rewards",
        "action_counts",
        "action_values",
    )

    @guard_agent_init
    def __init__(self, n_actions: int, epsilon: float, seed: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed=seed) if rng is None else rng

        self.step: Optional[int] = None
        self.action: Optional[int] = None
        self.rewards: Optional[List[float]] = None
        self.action_counts: Optional[np.ndarray] = None
        self.action_values: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.step = 0
        self.rewards = []
        self.action_counts = np.zeros(shape=(self.n_actions,), dtype=np.int32)
        self.action_values = np.zeros(shape=(self.n_actions,), dtype=np.float64)

    @guard_agent_act
    def act(self) -> int:
        self.step += 1

        if self.rng.random() < self.epsilon:
            self.action = self.rng.integers(self.n_actions).item()
        else:
            max_action_value = np.max(self.action_values).item()
            actions = np.flatnonzero(self.action_values == max_action_value)
            self.action = self.rng.choice(actions)

        return self.action

    @guard_agent_update
    def update(self, reward: float) -> None:
        self.rewards.append(reward)

        action = self.action
        self.action_counts[action] += 1
        self.action_values[action] += (reward - self.action_values[action]) / self.action_counts[action]


class BanditStepSizeAgent(BanditAgent):
    __slots__ = ("step_size",)

    @guard_agent_init
    def __init__(self, n_actions: int, epsilon: float, step_size: float, seed: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(n_actions=n_actions, epsilon=epsilon, seed=seed, rng=rng)
        self.step_size = step_size

    @guard_agent_update
    def update(self, reward: float) -> None:
        self.rewards.append(reward)

        action = self.action
        self.action_counts[action] += 1
        self.action_values[action] += self.step_size * (reward - self.action_values[action])


class BanditOptimisticAgent(BanditStepSizeAgent):
    __slots__ = ("initial_action_value",)

    def __init__(self, n_actions: int, epsilon: float, step_size: float, initial_action_value: float,
                 seed: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(n_actions=n_actions, epsilon=epsilon, step_size=step_size, seed=seed, rng=rng)
        self.initial_action_value = initial_action_value

    def reset(self) -> None:
        self.step = 0
        self.rewards = []
        self.action_counts = np.zeros(shape=(self.n_actions,), dtype=np.int32)
        self.action_values = np.full(shape=(self.n_actions,), fill_value=self.initial_action_value, dtype=np.float64)


class BanditUCBAgent(BanditAgent):
    __slots__ = ("c",)

    def __init__(self, n_actions: int, c: float, seed: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(n_actions=n_actions, epsilon=0.0, seed=seed, rng=rng)
        self.c = c

    @guard_agent_act
    def act(self) -> int:
        self.step += 1

        # Untried actions get +∞ so they’re picked first
        q = self.action_values + self.c * np.sqrt(np.log(self.step) / (self.action_counts + 1e-5))

        actions = np.flatnonzero(np.isclose(q, q.max()))
        self.action = self.rng.choice(actions)

        return self.action
