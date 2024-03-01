from src.infrastructure.attributes_repository import AttributesRepository
from dataclasses import dataclass, field
from configs import settings
from pathlib import Path
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
    def gets_best_model_checkpoint(self, model_name: str) -> Path:
        existing_models = list(
            filter(
                lambda path: path.match(f"*{model_name}*"),
                settings.EXISTING_MODEL_CHECKPOINT,
            )
        )
        existing_models = sorted(
            existing_models, key=lambda path: path.name.split("-")[-1]
        )
        latest_model_path = existing_models[0]
        print(latest_model_path.name)
        return latest_model_path
