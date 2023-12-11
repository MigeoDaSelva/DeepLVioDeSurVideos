from src.data_handler.abstract_strategies.abstract_id_generator import IDGenerator
from typing import Dict, List


class SequentialIDGenerator(IDGenerator):
    def generates(self, class_names: List[str]) -> Dict:
        return dict(zip(class_names, list(range(len(class_names)))))
