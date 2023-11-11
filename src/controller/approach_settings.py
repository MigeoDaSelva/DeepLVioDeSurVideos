from ipywidgets import (
    Dropdown,
    IntSlider,
    FloatLogSlider,
    FloatSlider,
    VBox,
    Label,
    Accordion,
    Tab,
    Checkbox,
    jslink,
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    CSVLogger,
    EarlyStopping,
    LearningRateScheduler,
)
from src.controller.setting_controller import SettingController
from src.domain.approach.abstract_approach import Approach
from src.domain.approach.conv2plus1d import Conv2Plus1D
from src.domain.approach.c3d_lstm import C3DLSTM
from src.domain.approach.movinet import Movinet
from dataclasses import dataclass, field
from livelossplot import PlotLossesKeras
from src.domain.approach.c3d import C3D
from IPython.display import display
from configs import settings
from datetime import date
import tensorflow as tf


@dataclass
class ApproachSettings(SettingController):
    load_model_container: VBox = field(
        init=False,
    )

    load_latest_model_checkbox: Checkbox = field(
        init=False,
    )

    unfreezing_checkbox: Checkbox = field(
        init=False,
    )

    approach_dropdown: Dropdown = field(
        init=False,
    )
    optimizer_dropdown: Dropdown = field(
        init=False,
    )
    padding_dropdown: Dropdown = field(
        init=False,
    )
    activation_dropdown: Dropdown = field(
        init=False,
    )
    epochs_slider: IntSlider = field(
        init=False,
    )
    filters_slider: FloatLogSlider = field(
        init=False,
    )
    dropout_slider: FloatSlider = field(
        init=False,
    )
    learning_rate_slider: FloatLogSlider = field(
        init=False,
    )

    kernel_size_container: VBox = field(
        init=False,
    )

    pooling_strides_container: VBox = field(
        init=False,
    )

    callbacks_accordion: Accordion = field(
        init=False,
    )

    @property
    def load_latest_model(self) -> bool:
        return self.load_latest_model_checkbox.value

    @property
    def unfreezing(self) -> bool:
        return self.unfreezing_checkbox.value

    @property
    def approach(self) -> Approach:
        return self.approach_dropdown.value

    @property
    def optimizer(self) -> str:
        return self.optimizer_dropdown.value

    @property
    def padding(self) -> str:
        return self.padding_dropdown.value

    @property
    def activation(self) -> str:
        return self.activation_dropdown.value

    @property
    def epochs(self) -> int:
        return self.epochs_slider.value

    @property
    def filters(self) -> int:
        return int(self.filters_slider.value)

    @property
    def dropout(self) -> float:
        return self.dropout_slider.value

    @property
    def learning_rate(self) -> float:
        return float("{:.1g}".format(self.learning_rate_slider.value))

    @property
    def kernel_size(self) -> tuple:
        return (
            self.kernel_size_container.children[1].value,
            self.kernel_size_container.children[2].value,
            self.kernel_size_container.children[3].value,
        )

    @property
    def pooling_strides(self) -> tuple:
        return (
            self.pooling_strides_container.children[1].value,
            self.pooling_strides_container.children[2].value,
            self.pooling_strides_container.children[3].value,
        )

    @property
    def callbacks(self) -> list:
        callbacks = []
        if not self.callbacks_accordion.children[0].children[0].children[0].value:
            callbacks.append(
                ModelCheckpoint(
                    filepath=f"{settings.MODEL_CHECKPOINT_PATH}/({date.today()})-{self.approach_dropdown.value.__name__}-model-"
                    + "{epoch:003d}-{val_loss:.5f}.keras"
                )
            )

        if not self.callbacks_accordion.children[0].children[1].children[0].value:
            callbacks.append(
                CSVLogger(
                    f"{settings.MODEL_TRAINING_HISTORY_PATH}/{self.approach_dropdown.value.__name__}-model-training-history.csv",
                    append=self.callbacks_accordion.children[0]
                    .children[1]
                    .children[1]
                    .value,
                )
            )
        if not self.callbacks_accordion.children[0].children[2].children[0].value:
            callbacks.append(
                EarlyStopping(
                    monitor=self.callbacks_accordion.children[0]
                    .children[2]
                    .children[1]
                    .value,
                    patience=self.callbacks_accordion.children[0]
                    .children[2]
                    .children[2]
                    .value,
                )
            )
        if not self.callbacks_accordion.children[0].children[3].children[0].value:
            callbacks.append(LearningRateScheduler(self._scheduler))

        if not self.callbacks_accordion.children[0].children[4].children[0].value:
            callbacks.append(PlotLossesKeras())
        return callbacks

    def show(self) -> None:
        display(
            self.load_model_container,
            self.approach_dropdown,
            self.optimizer_dropdown,
            self.activation_dropdown,
            self.padding_dropdown,
            self.epochs_slider,
            self.filters_slider,
            self.dropout_slider,
            self.learning_rate_slider,
            self.kernel_size_container,
            self.pooling_strides_container,
            self.callbacks_accordion,
        )

    def _on_value_change(self, change: dict) -> None:
        self.unfreezing_checkbox.disabled = not change["new"]

    def build(self) -> None:
        self.load_latest_model_checkbox = Checkbox(
            value=False,
            description="Load last model checkpoint",
            disabled=False,
            indent=False,
        )
        self.unfreezing_checkbox = Checkbox(
            value=False,
            description="Unfreezing",
            disabled=True,
        )

        self.load_latest_model_checkbox.observe(self._on_value_change, names="value")

        self.load_model_container = VBox(
            children=[
                self.load_latest_model_checkbox,
                self.unfreezing_checkbox,
            ],
            disabled=False,
        )

        self.approach_dropdown = Dropdown(
            options=[
                ("(2+1)D-CNN", Conv2Plus1D),
                ("C3D", C3D),
                ("C3D+LSTM", C3DLSTM),
                ("Movinet", Movinet),
            ],
            value=Conv2Plus1D,
            description="Approach options: ",
            disabled=False,
            layout=self.layout,
            style=self.style,
        )

        self.optimizer_dropdown = Dropdown(
            options=[
                "SGD",
                "RMSprop",
                "Adam",
                "AdamW",
                "Adadelta",
                "Adagrad",
                "Adamax",
                "Adafactor",
                "Nadam",
                "Ftrl",
            ],
            value=self.default_values.get("optimizer"),
            description="Optimizer options: ",
            disabled=False,
            layout=self.layout,
            style=self.style,
        )

        self.padding_dropdown = Dropdown(
            options=["same", "valid"],
            value=self.default_values.get("padding"),
            description="Padding options: ",
            disabled=False,
            layout=self.layout,
            style=self.style,
        )

        self.activation_dropdown = Dropdown(
            options=[
                "relu",
                "sigmoid",
                "softmax",
                "softplus",
                "softsign",
                "tanh",
                "selu",
                "elu",
                "exponential",
            ],
            value=self.default_values.get("activation"),
            description="Activation options: ",
            disabled=False,
            layout=self.layout,
            style=self.style,
        )

        self.epochs_slider = IntSlider(
            value=self.default_values.get("epochs"),
            min=1,
            max=300,
            step=1,
            description="Epochs: ",
            layout=self.layout,
            style=self.style,
        )

        self.filters_slider = FloatLogSlider(
            value=self.default_values.get("filters"),
            base=2,
            min=1,
            max=10,
            step=1,
            description="Filters: ",
            layout=self.layout,
            style=self.style,
        )

        self.dropout_slider = FloatSlider(
            value=self.default_values.get("dropout"),
            min=0.01,
            max=1.0,
            step=0.01,
            description="Dropout: ",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
            layout=self.layout,
            style=self.style,
        )

        self.learning_rate_slider = FloatLogSlider(
            value=self.default_values.get("learning_rate"),
            base=10,
            min=-7,
            max=-1,
            step=1,
            description="Learning rate: ",
            readout_format=".1g",
            layout=self.layout,
            style=self.style,
        )

        self.kernel_size_container = VBox(
            [
                Label(value="kernel Size Values:"),
                IntSlider(
                    value=self.default_values.get("kernel_size")[0],
                    min=1,
                    max=10,
                    step=1,
                    description="value 0: ",
                    layout=self.layout,
                    style=self.style,
                ),
                IntSlider(
                    value=self.default_values.get("kernel_size")[1],
                    min=1,
                    max=10,
                    step=1,
                    description="value 1: ",
                    layout=self.layout,
                    style=self.style,
                ),
                IntSlider(
                    value=self.default_values.get("kernel_size")[2],
                    min=1,
                    max=10,
                    step=1,
                    description="value 2: ",
                    layout=self.layout,
                    style=self.style,
                ),
            ],
            style=self.style,
        )

        self.pooling_strides_container = VBox(
            [
                Label(value="Pooling Strides Values:"),
                IntSlider(
                    value=self.default_values.get("pooling_strides")[0],
                    min=1,
                    max=10,
                    step=1,
                    description="value 0: ",
                    layout=self.layout,
                    style=self.style,
                ),
                IntSlider(
                    value=self.default_values.get("pooling_strides")[1],
                    min=1,
                    max=10,
                    step=1,
                    description="value 1: ",
                    layout=self.layout,
                    style=self.style,
                ),
                IntSlider(
                    value=self.default_values.get("pooling_strides")[2],
                    min=1,
                    max=10,
                    step=1,
                    description="value 2: ",
                    layout=self.layout,
                    style=self.style,
                ),
            ],
            style=self.style,
        )

        self.callbacks_accordion = Accordion(
            children=[self._build_callback_tabs()], titles=("Callbacks",)
        )

    def _build_callback_tabs(self) -> Tab:
        model_checkpoint = VBox(
            children=[
                Checkbox(
                    value=False, description="Disabled", disabled=False, indent=False
                )
            ]
        )

        csv_logger_checkbox = Checkbox(
            value=True, description="Disabled", disabled=False, indent=False
        )
        csv_logger_append = Checkbox(
            value=True, description="Append", disabled=False, indent=False
        )
        jslink((csv_logger_checkbox, "value"), (csv_logger_append, "disabled"))
        csv_logger = VBox(children=[csv_logger_checkbox, csv_logger_append])

        early_stopping_checkbox = Checkbox(
            value=False, description="Disabled", disabled=False, indent=False
        )
        early_stopping_monitor = Dropdown(
            options=["val_loss", "loss"],
            value="loss",
            description="Monitor: ",
            disabled=False,
            layout=self.layout,
            style=self.style,
        )
        early_stopping_patience = IntSlider(
            min=0,
            max=100,
            step=1,
            description="Patience: ",
            layout=self.layout,
            style=self.style,
        )
        jslink((early_stopping_checkbox, "value"), (early_stopping_monitor, "disabled"))
        jslink(
            (early_stopping_checkbox, "value"), (early_stopping_patience, "disabled")
        )
        early_stopping = VBox(
            children=[
                early_stopping_checkbox,
                early_stopping_monitor,
                early_stopping_patience,
            ]
        )

        lr_scheduler_checkbox = Checkbox(
            value=True, description="Disabled", disabled=False, indent=False
        )
        lr_scheduler_start = IntSlider(
            value=4,
            min=0,
            max=100,
            step=1,
            description="Starts at: ",
            layout=self.layout,
            style=self.style,
        )
        lr_scheduler_every = IntSlider(
            value=5,
            min=0,
            max=50,
            step=1,
            description="Every:",
            layout=self.layout,
            style=self.style,
        )
        jslink((lr_scheduler_checkbox, "value"), (lr_scheduler_start, "disabled"))
        jslink((lr_scheduler_checkbox, "value"), (lr_scheduler_every, "disabled"))
        lr_scheduler = VBox(
            children=[lr_scheduler_checkbox, lr_scheduler_start, lr_scheduler_every]
        )

        plot_losses_keras = VBox(
            children=[
                Checkbox(
                    value=True, description="Disabled", disabled=False, indent=False
                )
            ]
        )

        return Tab(
            children=[
                model_checkpoint,
                csv_logger,
                early_stopping,
                lr_scheduler,
                plot_losses_keras,
            ],
            titles=[
                "Model Checkpoint",
                "CSV Logger",
                "Early Stopping",
                "Learning Rate Scheduler",
                "Plot Losses Keras",
            ],
        )

    def _scheduler(self, epoch, lr):
        if epoch > 4 and epoch % 5 == 0:
            return lr * tf.math.exp(-0.1)
        else:
            return lr
