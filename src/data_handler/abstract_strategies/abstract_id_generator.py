from abc import ABC, abstractmethod
from typing import List, Dict


class IDGenerator(ABC):
    @abstractmethod
    def generates(self, class_names: List[str]) -> Dict:
        pass
