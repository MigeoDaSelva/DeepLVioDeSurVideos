from src.infrastructure.attributes_repository import AttributesRepository
from src.controller.approach_settings import ApproachSettings
from dataclasses import dataclass, field
from tensorflow.keras import Model
from configs import settings
from tensorflow import keras
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

    def loads_latest_model_checkpoint(
        self, approach_settings: ApproachSettings
    ) -> Model:
        existing_models = list(
            filter(
                lambda path: path.match(f"*{approach_settings.approach.__name__}*"),
                settings.EXISTING_MODEL_CHECKPOINT_FILES,
            )
        )
        existing_models.sort()
        existing_models.reverse()
        latest_model = existing_models[0]
        return keras.saving.load_model(latest_model)
