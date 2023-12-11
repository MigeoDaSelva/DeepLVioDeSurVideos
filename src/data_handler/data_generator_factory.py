from src.data_handler.strategies.video_creator import (
    DecordVideoCreator,
    OpenCVVideoCreator,
)
from src.data_handler.strategies.data_normalizer import (
    DataNormalizerComposite,
    ResizeWithPadding,
)
from src.data_handler.strategies.class_names_finder import UniqueClassNamesFinder
from src.data_handler.strategies.file_path_finder import FilePathFinderByLoad
from src.data_handler.strategies.id_generator import SequentialIDGenerator
from src.data_handler.data_generator import DataGenerator
from src.controller.data_settings import DataSettings
from pathlib import Path


class DataGeneratorFactory:
    @classmethod
    def creates(self, path: Path, data_settings: DataSettings) -> DataGenerator:
        generator = DataGenerator(dataset_path=path)

        generator.class_names_finder = UniqueClassNamesFinder()
        generator.file_path_finder = FilePathFinderByLoad()
        composite = DataNormalizerComposite()
        composite.add(ResizeWithPadding())
        generator.data_normalizer = composite
        generator.video_creator = DecordVideoCreator(data_settings.n_frames)
        generator.id_generator = SequentialIDGenerator()
        return generator
