import numpy as np
import plotly.graph_objects as go

from rl.bandit.environment import KArmedBanditEnvironment


def plot_k_armed_bandit_reward_distribution(n_actions: int, n_steps: int, seed: int) -> None:
    environment = KArmedBanditEnvironment(n_actions=n_actions, seed=seed)
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
