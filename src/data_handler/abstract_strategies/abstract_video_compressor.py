from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from numpy import ndarray, empty


@dataclass
class VideoCompressor(ABC):
    target_frames: int = field(default=30)
    _current_frames_set: ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._current_frames_set = empty(self.target_frames, dtype=ndarray)

    @property
    def video_frames(self) -> ndarray:
        return self._current_frames_set

    @abstractmethod
    def compresses(self, video: ndarray) -> ndarray:
        pass
