from abc import ABC, abstractmethod


class Trainer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass
