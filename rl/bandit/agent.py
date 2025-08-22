from typing import Optional

import numpy as np


class BanditAgent:
    __slots__ = ("n_actions", "epsilon", "step_size", "initial_action_value", "c", "rng", "time_step", "action_counts",
                 "action_values",)

    def __init__(self, n_actions: int, epsilon: float = 0.0, step_size: Optional[float] = None,
                 initial_action_value: Optional[float] = None, c: Optional[float] = None,
                 rng: Optional[np.random.Generator] = None, seed: int = 123):
        if n_actions <= 0:
            raise ValueError("n_action must be positive integer.")
        if not 0.0 <= epsilon < 1:
            raise ValueError("epsilon must be in [0, 1).")
        if step_size is not None and not 0.0 <= step_size <= 1:
            raise ValueError("step_size must be in [0, 1].")

        self.n_actions = n_actions
        self.epsilon = epsilon
        self.step_size = step_size
        self.initial_action_value = initial_action_value
        self.c = c
        self.rng = np.random.default_rng(seed) if rng is None else rng

        self.time_step: Optional[int] = None
        self.action_counts: Optional[np.ndarray[int]] = None
        self.action_values: Optional[np.ndarray[float]] = None

    def reset(self) -> None:
        self.time_step = 0
        self.action_counts = np.zeros(shape=(self.n_actions,), dtype=np.int64)

        value = 0.0 if self.initial_action_value is None else self.initial_action_value
        self.action_values = np.full(shape=(self.n_actions,), fill_value=value, dtype=np.float64)

    def act(self) -> int:
        if self.time_step is None:
            raise RuntimeError("Agent is not ready. Call `reset()` before the first `act()`.")

        self.time_step += 1

        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.n_actions)

        q = self.action_values.copy()
        if self.c is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                uncertainty = np.sqrt(np.log(self.time_step) / self.action_counts)

            uncertainty[np.isnan(uncertainty)] = np.inf  # Untried actions get infinite uncertainty
            q += self.c * uncertainty

        actions = np.where(np.isclose(q, np.max(q)))[0]
        return self.rng.choice(actions)

    def update(self, action: int, reward: float) -> None:
        if not (0 <= action < self.n_actions):
            raise ValueError(f"action must be in [0, {self.n_actions - 1}].")

        if self.time_step is None:
            raise RuntimeError("Agent is not ready. Call `reset()` then `act()` before `update()`.")

        self.action_counts[action] += 1

        alpha = self.step_size if self.step_size else 1 / self.action_counts[action]
        self.action_values[action] += alpha * (reward - self.action_values[action])


class BanditGradientAgent:
    __slots__ = ("n_actions", "step_size", "baseline", "time_step", "action_preferences", "average_reward", "rng",)

    def __init__(self, n_actions: int, step_size: float, baseline: bool, rng: Optional[np.random.Generator] = None,
                 seed: int = 123) -> None:
        if n_actions <= 0:
            raise ValueError("n_action must be positive integer.")
        if step_size is not None and not 0.0 <= step_size <= 1:
            raise ValueError("step_size must be in [0, 1].")

        self.n_actions = n_actions
        self.step_size = step_size
        self.baseline = baseline
        self.rng = np.random.default_rng(seed) if rng is None else rng

        self.time_step: Optional[int] = None
        self.action_preferences: Optional[np.ndarray] = None
        self.average_reward: Optional[float] = None

    def reset(self) -> None:
        self.time_step = 0
        self.action_preferences = np.zeros(shape=(self.n_actions,), dtype=np.float64)
        self.average_reward = 0.0

    def act(self) -> int:
        if self.time_step is None:
            raise RuntimeError("Agent is not ready. Call `reset()` before the first `act()`.")

        self.time_step += 1
        exp = np.exp(self.action_preferences - np.max(self.action_preferences))
        return self.rng.choice(self.n_actions, p=exp / np.sum(exp))

    def update(self, action: int, reward: float) -> None:
        if not (0 <= action < self.n_actions):
            raise ValueError(f"action must be in [0, {self.n_actions - 1}].")

        if self.time_step is None:
            raise RuntimeError("Agent is not ready. Call `reset()` then `act()` before `update()`.")

        exp = np.exp(self.action_preferences - np.max(self.action_preferences))
        probabilities = exp / np.sum(exp)
        self.action_preferences[action] += self.step_size \
                                           * (reward - self.average_reward) \
                                           * (1 - probabilities[action])
        mask = np.arange(self.n_actions) != action
        self.action_preferences[mask] -= self.step_size * (reward - self.average_reward) * probabilities[mask]

        if self.baseline:
            self.average_reward += (reward - self.average_reward) / self.time_step
