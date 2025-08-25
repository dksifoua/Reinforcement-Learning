import concurrent.futures
import copy
import os
from typing import List, Tuple, Union

import numpy as np
import tqdm

from rl.bandit.agent import BanditAgent, BanditAgentArguments, BanditGradientAgentArguments, BanditGradientAgent
from rl.bandit.environment import BanditEnvironment, BanditEnvironmentArguments, \
    BanditNonStationaryEnvironmentArguments, BanditNonStationaryEnvironment


def run_once(
        environment_args: Union[BanditEnvironmentArguments, BanditNonStationaryEnvironmentArguments],
        agent_args: Union[BanditAgentArguments, BanditGradientAgentArguments],
        agent_index: int,
        run_index: int,
        n_steps: int
) -> Tuple[
    int, int, np.ndarray, np.ndarray]:
    rewards = np.zeros(shape=(n_steps,), dtype=float)
    optimal_actions = np.zeros(shape=(n_steps,), dtype=bool)

    if type(environment_args) is BanditEnvironmentArguments:
        environment = BanditEnvironment(environment_args)
    elif type(environment_args) is BanditNonStationaryEnvironmentArguments:
        environment = BanditNonStationaryEnvironment(environment_args)
    else:
        raise ValueError("Environment arguments must be either BanditEnvironmentArguments or BanditNonStationaryEnvironmentArguments.")

    if type(agent_args) is BanditAgentArguments:
        agent = BanditAgent(agent_args)
    elif type(agent_args) is BanditGradientAgentArguments:
        agent = BanditGradientAgent(agent_args)
    else:
        raise ValueError("Agent arguments must be either BanditAgentArguments or BanditGradientAgentArguments.")

    environment.reset()
    agent.reset()
    for step in range(n_steps):
        action = agent.act()
        reward = environment.step(action=action)
        agent.update(action=action, reward=reward)

        rewards[step] = reward
        optimal_actions[step] = action == environment.best_action

    return agent_index, run_index, rewards, optimal_actions


class BanditExperiment:
    __slots__ = ("environment_args", "n_runs", "n_steps",)

    def __init__(
            self, environment_args: Union[BanditEnvironmentArguments, BanditNonStationaryEnvironmentArguments],
            n_steps: int,
            n_runs: int
    ) -> None:
        if n_runs < 1 or n_steps < 1:
            raise ValueError("n_runs and n_steps must be strictly positive.")

        self.environment_args = environment_args
        self.n_steps = n_steps
        self.n_runs = n_runs

    def start(
            self,
            agent_args_list: List[Union[BanditAgentArguments, BanditGradientAgentArguments]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not agent_args_list:
            raise ValueError("At least one agent params must be provided.")

        n_agents = len(agent_args_list)
        rewards = np.zeros(shape=(n_agents, self.n_runs, self.n_steps,), dtype=float)
        optimal_actions = np.zeros(shape=(n_agents, self.n_runs, self.n_steps,), dtype=bool)

        for agent_index, agent_args in enumerate(agent_args_list):
            with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = []
                for run_index in range(self.n_runs):
                    self.environment_args.seed = run_index
                    agent_args.seed = run_index
                    futures.append(executor.submit(run_once, copy.deepcopy(self.environment_args),
                                                   copy.deepcopy(agent_args), agent_index, run_index, self.n_steps))

                with tqdm.tqdm(total=self.n_runs, desc=f"Agent-{agent_index}") as pbar:
                    for completed_future in concurrent.futures.as_completed(futures):
                        _agent_index, _run_index, _rewards, _optimal_actions = completed_future.result()
                        rewards[_agent_index, _run_index] = _rewards
                        optimal_actions[_agent_index, _run_index] = _optimal_actions

                        pbar.update()

        return rewards.mean(axis=1), optimal_actions.mean(axis=1) * 100
