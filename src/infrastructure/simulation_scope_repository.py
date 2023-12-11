from src.infrastructure.attributes_repository import AttributesRepository
from dataclasses import dataclass, field
from configs import settings
import json


@dataclass
class SimulationScopeRepository(AttributesRepository):
    _attributes: dict = field(init=False)

    @property
    def attributes(self) -> dict:
        return self._attributes

    def read_attributes(self) -> None:
        with open(settings.SCOPE_SETTINGS_FILE) as file:
            attributes = json.load(file)

        existing_cross_validation = settings.EXISTING_CROSS_VALIDATION_FILES
        iterations = max(
            list(int(item.name.split("_")[0]) for item in existing_cross_validation)
        )
        folds = max(
            list(int(item.name.split("_")[1]) for item in existing_cross_validation)
        )
        attributes.update(
            {"existing_cross_validation": {"iterations": iterations, "folds": folds}}
        )
        self._attributes = attributes

    def save_attributes(self) -> None:
        pass

    def update_saved__attributes(self) -> None:
        pass
