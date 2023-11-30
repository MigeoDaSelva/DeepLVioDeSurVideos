from src.data_handler.abstract_strategies.abstract_file_path_finder import (
    FilePathFinder,
)
from dataclasses import dataclass
from configs import settings
from pathlib import Path
from typing import List
import pickle


@dataclass
class RecursiveFilePathFinder(FilePathFinder):
    file_extensions: List[str]

    def finds(self, path: Path) -> List[Path]:
        video_paths: List[Path] = []
        for extension in self.file_extensions:
            video_paths.extend(list(path.glob(f"*/*.{extension}")))
        return video_paths


@dataclass
class FilePathFinderByLoad(FilePathFinder):
    def finds(self, path: Path) -> List[Path]:
        with open(path, "rb") as file:
            data = pickle.load(file)
        return list(
            Path(f"{settings.DATASETS_PATH}/{settings.DATASET_NAME}/{path}")
            for path in list(data)
        )
