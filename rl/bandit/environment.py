import numpy as np
from typing import Optional


class KArmedBanditEnvironment:
    __slots__ = (
        "n_actions",
        "action_value_mean",
        "action_value_std",
        "reward_std",
        "rng",
        "action_values",
        "best_action"
    )

    def __init__(self, n_actions: int, action_value_mean: float = 0.0, action_value_std: float = 1.0,
                 reward_std: float = 1.0, seed: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if action_value_std < 0 or reward_std < 0:
            raise ValueError("stds must be non-negative")

        self.n_actions = n_actions
        self.action_value_mean = action_value_mean
        self.action_value_std = action_value_std
        self.reward_std = reward_std
        self.rng = np.random.default_rng(seed=seed) if rng is None else rng

        self.action_values: Optional[np.ndarray] = None
        self.best_action: Optional[int] = None

    def reset(self) -> None:
        self.action_values = self.rng.normal(
            loc=self.action_value_mean,
            scale=self.action_value_std,
            size=self.n_actions
        )
        self.best_action = np.argmax(self.action_values).item()

    def step(self, action: int) -> float:
        if self.best_action is None:
            raise RuntimeError("Environment is not ready. Call `reset()` before the first `step()`.")
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action must be between 0 and {self.n_actions - 1}")

        return self.rng.normal(
            loc=self.action_values[action].item(),
            scale=self.reward_std
        )
