from src.data_handler.dataset_factory import DatasetFactory
from src.domain.approach.abstract_approach import Approach
from src.data_handler.data_generator import DataGenerator
from dataclasses import dataclass, field
import tensorflow as tf


@dataclass
class Pipeline:
    approach: Approach
    train_dataset: DataGenerator
    validation_dataset: DataGenerator
    dataset_factory: DatasetFactory
    build_model: bool = field(default=True)

    _train_ds: tf.data.Dataset = field(init=False)
    _validation_ds: tf.data.Dataset = field(init=False)

    def __post_init__(self) -> None:
        self._train_ds: tf.data.Dataset = self.dataset_factory.creates(
            self.train_dataset
        )
        self._validation_ds: tf.data.Dataset = self.dataset_factory.creates(
            self.validation_dataset
        )

    def run(self) -> None:
        if self.build_model:
            self.approach.build()
        self.approach.compile()
        self.approach.fit(train_ds=self._train_ds, validation_ds=self._validation_ds)
