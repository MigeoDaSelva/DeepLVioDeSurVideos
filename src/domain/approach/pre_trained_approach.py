from src.domain.approach.abstract_approach import Approach
from dataclasses import dataclass, field
from abc import abstractmethod


@dataclass
class PreTrainedApproach(Approach):
    unfreezing: bool = field(default=False)

    @abstractmethod
    def build(self) -> None:
        pass
