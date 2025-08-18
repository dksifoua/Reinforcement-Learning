from typing import List, Optional

import numpy as np


class KArmedBanditAgent:
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

    def __init__(self, n_actions: int, epsilon: float, seed: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")

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
        self.action_values = np.empty(shape=(self.n_actions,), dtype=np.float64)

    def _choose_action(self) -> int:
        max_action_value = np.max(self.action_values).item()
        actions = np.flatnonzero(self.action_values == max_action_value)
        return self.rng.choice(actions)

    def act(self) -> int:
        if self.action_values is None:
            raise RuntimeError("Agent is not ready. Call `reset()` before the first `act()`.")

        self.step += 1
        if self.rng.random() < self.epsilon:
            self.action = self.rng.choice(self.n_actions)
        else:
            self.action = self._choose_action()
        return self.action

    def update(self, reward: float) -> None:
        self.rewards.append(reward)
        self.action_counts[self.action] += 1
        self.action_values[self.action] += (reward - self.action_values[self.action]) / self.action_counts[self.action]
