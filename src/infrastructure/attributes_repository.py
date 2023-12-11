from abc import ABC, abstractmethod


class AttributesRepository(ABC):
    @property
    @abstractmethod
    def attributes(self) -> dict:
        pass

    @abstractmethod
    def read_attributes(self) -> None:
        pass

    @abstractmethod
    def save_attributes(self) -> None:
        pass

    @abstractmethod
    def update_saved__attributes(self) -> None:
        pass
