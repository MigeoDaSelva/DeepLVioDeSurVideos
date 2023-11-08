from abc import ABC, abstractmethod
from ipywidgets import Layout


class WidgetsController(ABC):
    @property
    def layout(self) -> Layout:
        return Layout(
            width="auto",
            height="40px",
        )

    @property
    def style(self) -> dict:
        return {"description_width": "initial"}

    @abstractmethod
    def show(self) -> None:
        pass

    @abstractmethod
    def build(self) -> None:
        pass
