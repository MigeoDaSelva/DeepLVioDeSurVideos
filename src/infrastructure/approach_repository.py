from src.infrastructure.attributes_repository import AttributesRepository
from src.domain.approach.abstract_approach import Approach
from dataclasses import dataclass, field
from configs import settings
import json


@dataclass
class ApproachRepository(AttributesRepository):
    _attributes: dict = field(init=False)

    @property
    def attributes(self) -> dict:
        return self._attributes

    def read_attributes(self) -> None:
        with open(settings.APPROACH_SETTINGS_FILE) as file:
            attributes = json.load(file)
        self._attributes = attributes

    def save_attributes(self) -> None:
        pass

    def update_saved__attributes(self) -> None:
        pass

    @classmethod
    def loads_latest_model_checkpoint(self, approach: Approach) -> Approach:
        existing_models = list(
            filter(
                lambda path: path.match(f"*{approach.model.name}*"),
                settings.EXISTING_MODEL_CHECKPOINT,
            )
        )
        existing_models.sort()
        existing_models.reverse()
        latest_model_path = existing_models[0]
        print(latest_model_path.name)
        approach.model.load_weights(latest_model_path)
        return approach
