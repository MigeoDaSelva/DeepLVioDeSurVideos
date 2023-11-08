from src.controller.widgets_controller import WidgetsController
from dataclasses import dataclass


@dataclass
class SettingController(WidgetsController):
    default_values: dict

    def show(self) -> None:
        return super().show()

    def build(self) -> None:
        return super().build()
