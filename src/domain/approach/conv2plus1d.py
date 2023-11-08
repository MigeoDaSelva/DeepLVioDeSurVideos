from src.domain.approach.abstract_approach import Approach
from tensorflow import keras


class Conv2Plus1D(Approach):
    def build(self) -> None:
        _input = keras.layers.Input(shape=(self.input_shape[1:]))
        x = _input

        x = keras.layers.Conv3D(
            filters=self.filters,
            kernel_size=(1, 7, 7),
            padding=self.padding,
            activation=self.conv_activation,
        )(x)

        x = keras.layers.MaxPooling3D(
            pool_size=(1, 2, 2),
            strides=(1, self.pooling_strides[1], self.pooling_strides[2]),
            padding=self.padding,
        )(x)

        x = keras.layers.Conv3D(
            filters=self.filters,
            kernel_size=(self.kernel_size[0], 1, 1),
            padding=self.padding,
            activation=self.conv_activation,
        )(x)

        x = keras.layers.MaxPooling3D(
            strides=(self.pooling_strides[0], 1, 1), padding=self.padding
        )(x)

        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.GlobalAveragePooling3D()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(64)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.Dense(14)(x)
        x = keras.layers.Softmax()(x)

        self._model = keras.Model(_input, x, name="Conv2Plus1D")
