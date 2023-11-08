from src.data_handler.abstract_strategies.abstract_video_creator import VideoCreator
from dataclasses import dataclass, field
from numpy import ndarray, array
from decord import VideoReader
from cv2 import VideoCapture
from decord import cpu, gpu
from pathlib import Path
import cv2 as cv


@dataclass
class OpenCVVideoCreator(VideoCreator):
    """
    Factory Method
    """

    _reader: VideoCapture = field(init=False)

    @property
    def reader(self) -> VideoCapture:
        return self._reader

    def opens(self, video_path: Path) -> None:
        self._reader = VideoCapture(str(video_path))

    def extracts_frames(self) -> ndarray:
        frames: list = []

        if self.required_length:
            start, stop, step = self.calculates_range_values()
            for i in range(start, stop, step):
                self.reader.set(cv.CAP_PROP_POS_FRAMES, i)
                ret, frame = self.reader.read()
                frames.append(frame)
        else:
            ret, frame = self.reader.read()
            while ret:
                frames.append(frame)
                ret, frame = self.reader.read()

        return array(frames)[..., [2, 1, 0]]

    def gets_total_length(self) -> int:
        return int(self.reader.get(cv.CAP_PROP_FRAME_COUNT))


class DecordVideoCreator(VideoCreator):
    """
    Factory Method
    anable GPU... Build from source
    """

    _reader: VideoReader = field(init=False)

    @property
    def reader(self) -> VideoReader:
        return self._reader

    def opens(self, video_path: Path) -> None:
        self._reader = VideoReader(str(video_path), ctx=cpu(0))

    def extracts_frames(self) -> ndarray:
        start, stop, step = self.calculates_range_values()
        if self.required_length:
            frames = self.reader.get_batch(list(range(start, stop, step))).asnumpy()
        else:
            frames = self.reader.get_batch(list(range(stop))).asnumpy()
        return frames

    def gets_total_length(self) -> int:
        return len(self.reader)
