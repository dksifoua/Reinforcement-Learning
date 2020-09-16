import abc


class Environment(abc.ABC):

    @abc.abstractmethod
    def reset(self, seed=78):
        raise NotImplementedError

    @abc.abstractmethod
    def start(self):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action):
        raise NotImplementedError
