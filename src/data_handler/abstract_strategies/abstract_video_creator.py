from src.data_handler.abstract_strategies.abstract_class_names_finder import (
    ClassNamesFinder,
)
from src.data_handler.strategies.class_names_finder import UniqueClassNamesFinder
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from src.domain.video import Video
from numpy import ndarray
from pathlib import Path
from typing import Union


@dataclass
class VideoCreator(ABC):
    """
    Template Method and Strategy
    """

    video_path: Path = field(init=False)
    required_length: Union[None, int] = field(default=None)
    class_name_finder: ClassNamesFinder = field(default=UniqueClassNamesFinder())

    def creates(self, video_path: Path) -> Video:
        self.video_path = video_path
        new_video = Video()
        self.opens(self.video_path)
        new_video.frames = self.extracts_frames()
        new_video.label = self.gets_label()
        new_video.path = self.video_path
        new_video.name = self.video_path.name
        return new_video

    @abstractmethod
    def opens(self, video_path: Path) -> None:
        pass

    @abstractmethod
    def extracts_frames(self) -> ndarray:
        pass

    def gets_label(self) -> str:
        return self.class_name_finder.finds(file_paths=[self.video_path])[0]

    @abstractmethod
    def gets_total_length(self) -> int:
        pass

    def calculates_range_values(self) -> tuple:
        total_length = self.gets_total_length()
        step = int(total_length / self.required_length)
        stop = step * self.required_length
        start = 0 if stop <= total_length else step
        return start, stop, step
