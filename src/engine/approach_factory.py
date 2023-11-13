from tensorflow.keras.losses import (
    SparseCategoricalCrossentropy,
    BinaryCrossentropy,
    CategoricalCrossentropy,
)
from src.controller.approach_settings import ApproachSettings
from src.domain.approach.abstract_approach import Approach
from src.controller.data_settings import DataSettings


class ApproachFactory:
    @classmethod
    def creates(
        self, data_settings: DataSettings, approach_settings: ApproachSettings
    ) -> Approach:
        return approach_settings.approach(
            epochs=approach_settings.epochs,
            filters=approach_settings.filters,
            dropout=approach_settings.dropout,
            padding=approach_settings.padding,
            callbacks=approach_settings.callbacks,
            input_shape=(
                None,
                data_settings.n_frames,
                data_settings.image_height,
                data_settings.image_width,
                data_settings.n_channels,
            ),
            kernel_size=approach_settings.kernel_size,
            conv_activation=approach_settings.activation,
            pooling_strides=approach_settings.pooling_strides,
            loss=BinaryCrossentropy(
                # from_logits=True,
            ),  # when don't use softmax from_logits=True
            optimizer=approach_settings.optimizer,
        )
