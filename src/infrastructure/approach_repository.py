from src.infrastructure.attributes_repository import AttributesRepository
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
