from src.data_handler.abstract_strategies.abstract_class_names_finder import (
    ClassNamesFinder,
)
from src.data_handler.abstract_strategies.abstract_video_compressor import (
    VideoCompressor,
)
from src.data_handler.abstract_strategies.abstract_file_path_finder import (
    FilePathFinder,
)
from src.data_handler.abstract_strategies.abstract_data_normalizer import DataNormalizer
from src.data_handler.abstract_strategies.abstract_data_augmenter import DataAugmenter
from src.data_handler.abstract_strategies.abstract_video_creator import VideoCreator
from src.data_handler.abstract_strategies.abstract_id_generator import IDGenerator
from dataclasses import dataclass, field
from typing import Dict, List
from random import shuffle
from pathlib import Path


@dataclass
class DataGenerator:
    dataset_path: Path
    shuffle: bool = field(default=True)
    _class_names_finder: ClassNamesFinder = field(init=False)
    _video_compressor: VideoCompressor = field(init=False)
    _file_path_finder: FilePathFinder = field(init=False)
    _data_normalizer: DataNormalizer = field(init=False)
    _data_augmenter: DataAugmenter = field(init=False)
    _video_creator: VideoCreator = field(init=False)
    _id_generator: IDGenerator = field(init=False)

    _class_names_and_ids: Dict = field(init=False)
    _file_paths: List[Path] = field(init=False)

    def __call__(self):
        self._file_paths = self._file_path_finder.finds(path=self.dataset_path)
        class_names = self._class_names_finder.finds(file_paths=self.file_paths)
        self._class_names_and_ids = self._id_generator.generates(
            class_names=class_names
        )

        if self.shuffle:
            shuffle(self.file_paths)

        for file_path in self.file_paths:
            video = self._video_creator.creates(video_path=file_path)
            # self._data_augmenter.augments(video=video)
            # self._video_compressor.compresses(video=video)
            self._data_normalizer.normalizes(video=video)
            class_name = self._class_names_finder.finds(file_paths=[file_path])[0]
            label = self.class_names_and_ids[class_name]
            yield video.frames, label

    @property
    def file_paths(self) -> List[Path]:
        return self._file_paths

    @property
    def class_names_and_ids(self) -> Dict:
        return self._class_names_and_ids

    @property
    def file_path_finder(self) -> FilePathFinder:
        return self._file_path_finder

    @file_path_finder.setter
    def file_path_finder(self, strategy: FilePathFinder) -> None:
        self._file_path_finder = strategy

    @property
    def class_names_finder(self) -> ClassNamesFinder:
        return self._class_names_finder

    @class_names_finder.setter
    def class_names_finder(self, strategy: ClassNamesFinder) -> None:
        self._class_names_finder = strategy

    @property
    def id_generator(self) -> IDGenerator:
        return self._id_generator

    @id_generator.setter
    def id_generator(self, strategy: IDGenerator) -> None:
        self._id_generator = strategy

    @property
    def video_creator(self) -> VideoCreator:
        return self._video_creator

    @video_creator.setter
    def video_creator(self, strategy: VideoCreator) -> None:
        self._video_creator = strategy

    @property
    def video_compressor(self) -> VideoCompressor:
        return self._video_compressor

    @video_compressor.setter
    def video_compressor(self, strategy: VideoCompressor) -> None:
        self._video_compressor = strategy

    @property
    def data_normalizer(self) -> DataNormalizer:
        return self._data_normalizer

    @data_normalizer.setter
    def data_normalizer(self, strategy: DataNormalizer) -> None:
        self._data_normalizer = strategy

    @property
    def data_augmenter(self) -> DataAugmenter:
        return self._data_augmenter

    @data_augmenter.setter
    def data_augmenter(self, strategy: DataAugmenter) -> None:
        self._data_augmenter = strategy
