import numpy as np
import plotly.graph_objects as go
from typing import Optional


class KArmedBanditEnvironment:

    def __init__(self, n_actions: int, seed: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> None:
        if n_actions <= 0:
            raise ValueError("n_actions must be a positive integer.")

        self.n_actions = n_actions
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        self.action_values: Optional[np.ndarray] = None
        self.best_action: Optional[int] = None
        self.reward_noise_std: Optional[float] = None

    def reset(self, q_mean: float = 0.0, q_std: float = 1.0, reward_noise_std: float = 1.0) -> None:
        self.action_values = self.rng.normal(loc=q_mean, scale=q_std, size=self.n_actions)
        self.best_action = np.argmax(self.action_values).item()
        self.reward_noise_std = reward_noise_std

    def step(self, action: int) -> float:
        if self.action_values is None:
            raise RuntimeError("Environment not ready. Call `reset()` before the first `step()`.")

        if not (0 <= action < self.n_actions):
            raise IndexError(f"action must be in [0, {self.n_actions - 1}], got {action}!")

        return self.rng.normal(loc=self.action_values[action], scale=self.reward_noise_std).item()

    @staticmethod
    def plot_reward_distribution(n_actions: int = 10, n_steps: int = 1000, seed: int = 123) -> None:
        environment = KArmedBanditEnvironment(n_actions=n_actions, seed=seed)
        environment.reset()

        rewards = environment.rng.normal(
            loc=environment.action_values[:, None],
            scale=environment.reward_noise_std,
            size=(n_actions, n_steps),
        )
        mean_rewards = rewards.mean(axis=1)

        fig = go.Figure()
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        tick_half_width = 0.3
        for action in range(environment.n_actions):
            fig.add_trace(go.Violin(
                x=action + np.zeros(n_steps,),
                y=rewards[action],
                name=f"Action {action}",
                points=False,
                scalemode="count",
            ))
            fig.add_trace(go.Scatter(
                x=[action - tick_half_width, action + tick_half_width],
                y=np.zeros(2,) + mean_rewards[action],
                mode="lines",
                line=dict(color="gray", width=2),
                hovertemplate=f"q*({action}) = {mean_rewards[action]:.3f}<extra></extra>",
                showlegend=False
            ))
            fig.add_annotation(
                x=action + tick_half_width + 0.05, y=mean_rewards[action],
                text=f"q*({action})",
                showarrow=False, xanchor="left", yanchor="middle",
                font=dict(color="gray")
            )

        fig.update_layout(
            template="plotly_white",
            xaxis=dict(
                title="Action",
                tickmode="array",
                tickvals=[*range(environment.n_actions)]
            ),
            yaxis=dict(title="Reward distribution"),
            violingap=0.15,
            margin=dict(l=50, r=30, t=20, b=40),
            height=480
        )
        fig.show()