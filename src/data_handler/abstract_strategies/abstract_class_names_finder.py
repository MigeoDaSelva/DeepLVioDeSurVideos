from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class ClassNamesFinder(ABC):
    @abstractmethod
    def finds(self, file_paths: List[Path]) -> List[str]:
        pass
