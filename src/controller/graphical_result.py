from src.controller.result_controller import ResultController
from dataclasses import dataclass, field
from IPython.display import display
from ipywidgets import Image


@dataclass
class GraphicalResult(ResultController):
    image_as_bytes: bytes

    image: Image = field(
        init=False,
    )

    def show(self) -> None:
        display(self.image)

    def build(self) -> None:
        self.image = Image(value=self.image_as_bytes, format="png")
