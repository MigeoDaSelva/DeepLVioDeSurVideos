from src.data_handler.abstract_strategies.abstract_class_names_finder import (
    ClassNamesFinder,
)
from pathlib import Path
from typing import List


class UniqueClassNamesFinder(ClassNamesFinder):
    def finds(self, file_paths: List[Path]) -> List[str]:
        classes = list({path.parent.name for path in file_paths})
        classes.sort()
        return classes


class ClassNamesSequentialFinder(ClassNamesFinder):
    def finds(self, file_paths: List[Path]) -> List[str]:
        classes = list(path.parent.name for path in file_paths)
        return classes
