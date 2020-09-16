from rl.agents.base import Agent
from rl.environments.base import Environment


class RLGlue:

    def __init__(self, env: Environment, agent: Agent):
        self.env = env
        self.agent = agent
        self.total_reward = 0.
        self.last_action = 0
        self.num_steps = 0
        self.num_episodes = 0

    def start(self):
        last_state = self.env.start()
        self.last_action = self.agent.start(last_state)
        return last_state, self.last_action

    def step(self):
        reward, observation, done, _ = self.env.step(self.last_action)
        self.total_reward += reward
        if done:
            self.num_episodes += 1
            self.agent.end(reward, observation)
        else:
            self.num_steps += 1
            self.last_action = self.agent.step(reward, observation)
        return reward, observation, self.last_action, done

    def reset(self, seed=78):
        self.env.reset(seed=seed)
        self.agent.reset()
