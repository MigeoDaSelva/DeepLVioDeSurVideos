from src.data_handler.dataset_factory import DatasetFactory
from src.domain.approach.abstract_approach import Approach
from src.data_handler.data_generator import DataGenerator
from dataclasses import dataclass
import tensorflow as tf


@dataclass
class Pipeline:
    approach: Approach
    train_dataset: DataGenerator
    validation_dataset: DataGenerator
    dataset_factory: DatasetFactory

    def run(self) -> None:
        train_ds: tf.data.Dataset = self.dataset_factory.creates(self.train_dataset)
        validation_ds: tf.data.Dataset = self.dataset_factory.creates(
            self.validation_dataset
        )
        self.approach.build()
        self.approach.compile()
        self.approach.fit(train_ds=train_ds, validation_ds=validation_ds)
