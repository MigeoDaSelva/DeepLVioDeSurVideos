from src.controller.setting_controller import SettingController
from dataclasses import dataclass, field
from IPython.display import display
from ipywidgets import IntSlider


@dataclass
class DataSettings(SettingController):
    batch_size_slider: IntSlider = field(
        init=False,
    )
    n_frames_slider: IntSlider = field(
        init=False,
    )
    height_slider: IntSlider = field(
        init=False,
    )
    width_slider: IntSlider = field(
        init=False,
    )
    n_channels_slider: IntSlider = field(
        init=False,
    )

    @property
    def batch_size(self) -> int:
        return self.batch_size_slider.value

    @property
    def n_frames(self) -> int:
        return self.n_frames_slider.value

    @property
    def image_height(self) -> int:
        return self.height_slider.value

    @property
    def image_width(self) -> int:
        return self.width_slider.value

    @property
    def n_channels(self) -> int:
        return self.n_channels_slider.value

    def show(self) -> None:
        display(
            self.batch_size_slider,
            self.n_frames_slider,
            self.height_slider,
            self.width_slider,
            self.n_channels_slider,
        )

    def build(self) -> None:
        self.batch_size_slider = IntSlider(
            value=self.default_values.get("batch_size"),
            min=1,
            max=1000,
            step=1,
            description="Batch size: ",
            layout=self.layout,
            style=self.style,
        )
        self.n_frames_slider = IntSlider(
            value=self.default_values.get("n_frames"),
            min=1,
            max=100,
            step=1,
            description="Number of frames: ",
            layout=self.layout,
            style=self.style,
        )
        self.height_slider = IntSlider(
            value=self.default_values.get("height"),
            min=1,
            max=2000,
            step=1,
            description="Image height: ",
            layout=self.layout,
            style=self.style,
        )
        self.width_slider = IntSlider(
            value=self.default_values.get("width"),
            min=1,
            max=2000,
            step=1,
            description="Image width: ",
            layout=self.layout,
            style=self.style,
        )
        self.n_channels_slider = IntSlider(
            value=self.default_values.get("n_channels"),
            min=1,
            max=10,
            step=1,
            description="Number of channels: ",
            layout=self.layout,
            style=self.style,
        )
