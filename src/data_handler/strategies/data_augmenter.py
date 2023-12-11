from src.data_handler.abstract_strategies.abstract_data_augmenter import DataAugmenter
from numpy import ndarray, fliplr, flipud, empty, rot90, append
from dataclasses import dataclass, field
from src.domain.video import Video
from random import randint
from typing import List


@dataclass
class DataAugmenterComposite(DataAugmenter):
    children: List[DataAugmenter]

    def add(self, children: DataAugmenter) -> None:
        self.children.append(children)

    def remove(self, children: DataAugmenter) -> None:
        self.children.remove(children)

    def augments(self, video: Video) -> None:
        for data_augmenter in self.children:
            video = data_augmenter.augments(video)
        return video


@dataclass
class HorizontalFlip(DataAugmenter):
    odds: int = field(default=50)

    def augments(self, video: Video) -> None:
        new_video_frames = empty(shape=video.frames.shape)
        for i, frame in enumerate(video.frames):
            shape = frame.shape
            new_frame = empty(shape=(1, shape[0], shape[1], shape[2]))
            if self.odds > randint(1, 100):
                first_part = new_video_frames[:i]
                first_part.append(fliplr(frame))
                new_video_frames.extend(first_part)
                second_part = video_frames[i : len(video_frames)]
                new_video_frames.extend(second_part)
        return new_video_frames


@dataclass
class VerticalFlip(DataAugmenter):
    odds: int = field(default=50)

    def augments(self, video_frames: List[ndarray]) -> List[ndarray]:
        new_video_frames: List[ndarray] = []
        for i, frame in enumerate(video_frames):
            if self.odds > randint(1, 100):
                first_part = video_frames[:i]
                first_part.append(flipud(frame))
                new_video_frames.extend(first_part)
                second_part = video_frames[i : len(video_frames)]
                new_video_frames.extend(second_part)
        return new_video_frames
