import functools
import inspect
from collections.abc import Callable

import numpy as np
import plotly.graph_objects as go

from rl.bandit.agent import BanditAgent
from rl.bandit.environment import BanditEnvironment


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


def plot_bandit_reward_distribution(n_actions: int, n_steps: int, seed: int) -> None:
    environment = BanditEnvironment(n_actions=n_actions, seed=seed)
    environment.reset()

    rewards = environment.rng.normal(
        loc=environment.action_values[:, None],
        scale=environment.reward_std,
        size=(n_actions, n_steps)
    )

    average_rewards = rewards.mean(axis=1)

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    tick_half_width = 0.3

    for action in range(n_actions):
        fig.add_trace(
            trace=go.Violin(
                x=np.zeros(shape=(n_steps,)) + action,
                y=rewards[action],
                name=f"Action {action}",
                points=False,
                scalemode="count",
                showlegend=False,
            )
        )
        fig.add_trace(
            trace=go.Scatter(
                x=[action - tick_half_width, action + tick_half_width],
                y=np.zeros(shape=(2,)) + average_rewards[action],
                mode="lines",
                line=dict(color="gray", width=2),
                hovertemplate=f"q*({action}) = {average_rewards[action]:.3f}<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=action + tick_half_width + 0.05,
            y=average_rewards[action],
            text=f"q*({action})",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(color="gray"),
        )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Action"),
        yaxis=dict(title="Reward distribution"),
        violingap=0.15,
        margin=dict(l=50, r=30, t=20, b=40),
        width=1000,
        height=500
    )
    fig.show()
