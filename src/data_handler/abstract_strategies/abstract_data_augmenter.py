from abc import ABC, abstractmethod
from src.domain.video import Video


class DataAugmenter(ABC):
    @abstractmethod
    def augments(self, video: Video) -> None:
        pass
