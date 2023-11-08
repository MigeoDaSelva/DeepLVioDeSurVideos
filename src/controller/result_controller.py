from src.controller.widgets_controller import WidgetsController
from dataclasses import dataclass


@dataclass
class ResultController(WidgetsController):
    def show(self) -> None:
        return super().show()

    def build(self) -> None:
        return super().build()
