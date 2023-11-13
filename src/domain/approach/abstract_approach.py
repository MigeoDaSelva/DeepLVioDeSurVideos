from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from tensorflow import Tensor
import tensorflow as tf


@dataclass
class Approach(ABC):
    """Adapter pattern"""

    epochs: int
    filters: int
    dropout: float
    padding: tuple
    callbacks: list
    input_shape: tuple
    kernel_size: tuple
    conv_activation: str
    pooling_strides: tuple
    loss: tf.keras.losses.Loss
    optimizer: tf.keras.optimizers.Optimizer
    metrics: list = field(init=False)

    _model: tf.keras.Model = field(init=False)
    _history: tf.keras.callbacks.History = field(init=False)
    _actual: Tensor = field(init=False)
    _predicted: Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.metrics = [
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=1, name="top_1", dtype=tf.float32
            ),
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=5, name="top_5", dtype=tf.float32
            ),
            "accuracy",
        ]

    @abstractmethod
    def build(self) -> None:
        pass

    def compile(self) -> None:
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )

    def fit(self, train_ds: tf.data.Dataset, validation_ds: tf.data.Dataset) -> None:
        self._history = self.model.fit(
            x=train_ds,
            epochs=self.epochs,
            validation_data=validation_ds,
            callbacks=self.callbacks,
            use_multiprocessing=True,
            # verbose=0,
        )

    def evaluate(self, test_ds: tf.data.Dataset) -> None:
        self.model.evaluate(test_ds, return_dict=True)
        actual = [labels for _, labels in test_ds.unbatch()]
        predicted = self.model.predict(test_ds)

        actual = tf.stack(actual, axis=0)
        predicted = tf.concat(predicted, axis=0)
        predicted = tf.argmax(predicted, axis=1)
        self._actual = actual
        self._predicted = predicted

    @property
    def actual(self) -> Tensor:
        return self._actual

    @property
    def predicted(self) -> Tensor:
        return self._predicted

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    @property
    def history(self) -> tf.keras.callbacks.History:
        return self._history
