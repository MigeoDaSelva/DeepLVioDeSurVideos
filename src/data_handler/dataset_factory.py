from src.data_handler.data_generator import DataGenerator
from dataclasses import dataclass, field
import tensorflow as tf


@dataclass
class DatasetFactory:
    batch_size: int
    output_shape: tuple
    keep_cached: bool = field(default=True)
    _output_signature: tf.TensorSpec = field(init=False)

    def __post_init__(self) -> None:
        self._output_signature = (
            tf.TensorSpec(
                shape=self.output_shape,
                dtype=tf.float32,
            ),
            tf.TensorSpec(shape=(), dtype=tf.int16),
        )

    def creates(self, data_generator: DataGenerator) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(
            data_generator, output_signature=self._output_signature
        )
        if self.keep_cached:
            dataset = dataset.cache()
        if data_generator.shuffle:
            dataset = dataset.shuffle(buffer_size=dataset.cardinality())
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
