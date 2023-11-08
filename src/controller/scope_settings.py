from ipywidgets import Dropdown, IntSlider, VBox, IntRangeSlider, Stack, Label, jslink
from src.controller.setting_controller import SettingController
from dataclasses import dataclass, field
from IPython.display import display


@dataclass
class ScopeSettings(SettingController):
    iterations_v_box: VBox = field(
        init=False,
    )

    folds_v_box: VBox = field(
        init=False,
    )

    _caption_v_box: VBox = field(
        init=False,
    )

    @property
    def iterations(self) -> list:
        option = self.iterations_v_box.children[1].value
        if option == "Range":
            return list(
                range(
                    self.iterations_v_box.children[2].children[2].value[0],
                    self.iterations_v_box.children[2].children[2].value[1] + 1,
                )
            )
        elif option == "Sequential":
            return list(range(self.iterations_v_box.children[2].children[1].value + 1))
        else:
            return [self.iterations_v_box.children[2].children[0].value]

    @property
    def folds(self) -> list:
        option = self.folds_v_box.children[1].value
        if option == "Range":
            return list(
                range(
                    self.folds_v_box.children[2].children[2].value[0],
                    self.folds_v_box.children[2].children[2].value[1] + 1,
                )
            )
        elif option == "Sequential":
            return list(range(self.folds_v_box.children[2].children[1].value + 1))
        else:
            return [self.folds_v_box.children[2].children[0].value]

    def show(self) -> None:
        display(self._caption_v_box, self.iterations_v_box, self.folds_v_box)

    def build(self) -> None:
        itens = [
            Dropdown(
                options=list(
                    range(
                        self.default_values.get("existing_cross_validation").get(
                            "iterations"
                        )
                        + 1
                    )
                ),
                description="Literally: ",
                disabled=False,
            ),
            IntSlider(
                min=0,
                max=self.default_values.get("existing_cross_validation").get(
                    "iterations"
                ),
                step=1,
                description="Sequential: ",
                disabled=False,
            ),
            IntRangeSlider(
                min=0,
                max=self.default_values.get("existing_cross_validation").get(
                    "iterations"
                ),
                step=1,
                description="Range",
                disabled=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            ),
        ]

        stack = Stack(itens, selected_index=0)
        dropdown = Dropdown(
            options=["Literally", "Sequential", "Range"],
        )
        description = Label(value="Select iteration(s):")
        jslink((dropdown, "index"), (stack, "selected_index"))
        self.iterations_v_box = VBox(
            [description, dropdown, stack],
            style=self.style,
            indent=False,
        )

        itens = [
            Dropdown(
                options=list(
                    range(
                        self.default_values.get("existing_cross_validation").get(
                            "folds"
                        )
                        + 1
                    )
                ),
                description="Literally: ",
                disabled=False,
            ),
            IntSlider(
                min=0,
                max=self.default_values.get("existing_cross_validation").get("folds"),
                step=1,
                description="Sequential: ",
                disabled=False,
            ),
            IntRangeSlider(
                min=0,
                max=self.default_values.get("existing_cross_validation").get("folds"),
                step=1,
                description="Range",
                disabled=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            ),
        ]

        stack = Stack(itens, selected_index=0)
        dropdown = Dropdown(
            options=["Literally", "Sequential", "Range"],
        )
        description = Label(value="Select fold(s):")
        jslink((dropdown, "index"), (stack, "selected_index"))
        self.folds_v_box = VBox(
            [description, dropdown, stack],
            style=self.style,
        )

        self._caption_v_box = VBox(
            children=[
                Label(
                    value="Literally: Is used to indicate the execution of only one specific iteration or fold. i.e: [a] = {x ∈ ℤ : x = a}"
                ),
                Label(
                    value="Sequential: Defines a sequence starting from the first existing iteration or fold up to exactly the indicated point a. Such as: [0, a] = {x ∈ ℤ : 0 ≤ x ≤ a}"
                ),
                Label(
                    value="Range: Executes a sequence that starts and ends at a chosen closed interval. as well as: [a,b] = { x ∈ ℤ : a ≤ x ≤ b}"
                ),
            ]
        )
