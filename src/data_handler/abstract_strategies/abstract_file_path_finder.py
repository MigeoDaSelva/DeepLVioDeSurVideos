from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class FilePathFinder(ABC):
    @abstractmethod
    def finds(self, path: Path) -> List[Path]:
        pass
