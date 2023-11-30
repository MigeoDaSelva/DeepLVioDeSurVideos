from src.data_handler.strategies.compressor_stop_criterias.abstract_stop_criteria import (
    StopCriteria,
)
from src.data_handler.strategies.abstract_id_generator import VideoCompressor
from functools import singledispatchmethod
from dataclasses import dataclass, field
from itertools import combinations
from collections import deque
from cv2 import VideoCapture
from numpy import ndarray
from typing import List
import cv2 as cv


@dataclass
class DeltaFrameBasedVideoCompressor(VideoCompressor):
    stop_criteria: StopCriteria
    frame_range: int = field(default=3)

    _last_frame_set: List[ndarray] = field(init=False)
    _initial_n_frames: int = field(init=False)

    def compresses(self, video: VideoCapture) -> List[ndarray]:
        self._selects_frames(video)
        while not self.stop_criteria.is_finished(self):
            self._selects_frames(self._current_frames_set)
        video.release()
        return self.video_frames

    @singledispatchmethod
    def _selects_frames(self, video) -> None:
        raise NotImplementedError(
            f"Method not yet implemented for this type: {type(video).__name__}"
        )

    @_selects_frames.register
    def _(self, video: VideoCapture) -> None:
        self._initial_n_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        limit: int = self._computes_limit()
        for _ in range(limit):
            if self.stop_criteria.is_finished(self):
                break
            current_frames: List[ndarray] = []
            for _ in range(self.frame_range):
                current_frames.append(video.read()[1])
            self._drops_frames(current_frames)

        is_finished, frame = video.read()
        while not is_finished:
            self._current_frames_set.append(frame)
            is_finished, frame = video.read()

    @_selects_frames.register
    def _(self, video: list) -> None:
        self._last_frame_set = video
        self._current_frames_set = []
        self._initial_n_frames = len(video)
        limit: int = self._computes_limit()
        frame_queue = deque(video)

        for _ in range(limit):
            if self.stop_criteria.is_finished(self):
                break
            current_frames: List[ndarray] = []
            for _ in range(self.frame_range):
                current_frames.append(frame_queue.popleft())
            self._drops_frames(current_frames)

        self._current_frames_set.extend(frame_queue)

    def _computes_limit(self) -> int:
        return (
            self._initial_n_frames
            if self._initial_n_frames % self.frame_range == 0
            else (self._initial_n_frames // self.frame_range) * self.frame_range
        )

    def _drops_frames(self, frames: List[ndarray]) -> None:
        frames_and_mean_dframe: List[list] = []
        frames_combinations = list(combinations(frames, 2))
        combinations_to_save = len(frames_combinations) // 2

        for combination in frames_combinations:
            delta_frame = cv.absdiff(combination[0], combination[1])
            mean_of_delta_frame = delta_frame.mean()
            frames_and_mean_dframe.append([combination, mean_of_delta_frame])

        frames_and_mean_dframe = sorted(frames_and_mean_dframe, key=lambda x: x[1])

        # Dicionar condições de exclusão
        for _ in range(combinations_to_save):
            self._current_frames_set.extend(list(frames_and_mean_dframe.pop()[0]))


@dataclass
class UniformVideoCompressor(VideoCompressor):
    def compresses(self, frames: List[ndarray]) -> List[ndarray]:
        step = int(len(frames) / self.target_frames)
        for i in range(0, len(frames), step):
            self._current_frames_set.append(frames[i])
        return self.video_frames
