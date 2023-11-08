from src.data_handler.abstract_strategies.abstract_data_normalizer import DataNormalizer
from dataclasses import dataclass, field
from src.domain.video import Video
from numpy import empty, float16
from typing import List
import tensorflow as tf


@dataclass
class DataNormalizerComposite(DataNormalizer):
    _children: List[DataNormalizer] = field(init=False)

    def __post_init__(self) -> None:
        self._children = []

    def add(self, children: DataNormalizer) -> None:
        self._children.append(children)

    def remove(self, children: DataNormalizer) -> None:
        self._children.remove(children)

    def normalizes(self, video: Video) -> None:
        for data_normalizer in self._children:
            data_normalizer.normalizes(video)


class ChangeImageType(DataNormalizer):
    def normalizes(self, video: Video) -> None:
        new_video_frames = empty(shape=video.frames.shape, dtype=float16)
        for i, frame in enumerate(video.frames):
            new_video_frames[i] = frame.astype(float16)
        video.frames = new_video_frames


@dataclass
class ResizeWithPadding(DataNormalizer):
    output_size: tuple = field(default=(224, 224))

    def normalizes(self, video: Video) -> None:
        new_video_frames = empty(
            shape=(
                video.frames.shape[0],
                self.output_size[0],
                self.output_size[1],
                video.frames.shape[3],
            ),
            dtype=float16,
        )
        for i, frame in enumerate(video.frames):
            new_video_frames[i] = tf.image.resize_with_pad(
                frame.astype(float16), *self.output_size
            )
        video.frames = new_video_frames
