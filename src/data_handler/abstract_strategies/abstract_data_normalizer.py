from abc import ABC, abstractmethod
from src.domain.video import Video


class DataNormalizer(ABC):
    @abstractmethod
    def normalizes(self, video: Video) -> None:
        pass
