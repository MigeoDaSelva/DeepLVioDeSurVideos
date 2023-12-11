from src.controller.result_controller import ResultController
from dataclasses import dataclass, field
from IPython.display import display
from ipywidgets import Output
from pandas import DataFrame


@dataclass
class MetricsResult(ResultController):
    metrics: DataFrame

    output: Output = field(
        init=False,
    )

    def show(self) -> None:
        with self.output:
            display(self.metrics)

    def build(self) -> None:
        self.output = Output()
