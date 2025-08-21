import functools
import inspect
from collections.abc import Callable

import numpy as np

from rl.bandit import BanditAgent


def guard_agent_init(func: Callable) -> Callable:
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> None:
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        n_actions = bound.arguments.get("n_actions")
        if not isinstance(n_actions, int) or n_actions <= 0:
            raise ValueError("n_actions must be a positive integer")

        epsilon = bound.arguments.get("epsilon")
        try:
            eps = float(epsilon)

            if not np.isfinite(eps) or not (0.0 <= eps <= 1.0):
                raise ValueError("epsilon must be between 0 and 1")
        except (TypeError, ValueError):
            raise TypeError("epsilon must be a real number in [0, 1]")

        rng = bound.arguments.get("rng")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator or None")

        step_size = bound.arguments.get("step_size")
        if step_size is not None and not 0 < step_size <= 1:
            raise ValueError("step_size must be between 0 and 1")

        return func(*args, **kwargs)

    return wrapper


def guard_agent_act(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self: BanditAgent) -> None:
        if self.action_values is None:
            raise RuntimeError("Agent is not ready. Call `reset()` before the first `act()`.")

        return func(self)

    return wrapper


def guard_agent_update(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self: BanditAgent, reward: float) -> None:
        if self.action_values is None:
            raise RuntimeError("Agent is not ready. Call `reset()` and `act()` before `update()`.")

        if self.action is None:
            raise RuntimeError("Agent did not act. Call act() before update().")

        if not (isinstance(reward, (int, float)) and np.isfinite(reward)):
            raise ValueError("reward must be a finite number.")

        return func(self, reward)

    return wrapper