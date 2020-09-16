import abc


class Agent(abc.ABC):

    @abc.abstractmethod
    def reset(self, **kwargs):
        pass

    @abc.abstractmethod
    def start(self, state, **kwargs):
        pass

    @abc.abstractmethod
    def step(self, reward, state, **kwargs):
        pass

    @abc.abstractmethod
    def end(self, reward, state, **kwargs):
        pass
