from src.domain.approach.abstract_approach import Approach
from tensorflow import keras


class C3D(Approach):
    def build(self) -> None:
        _input = keras.layers.Input(shape=(self.input_shape[1:]))
        x = _input
        filters = self.filters

        # 1# Conv
        x = keras.layers.Conv3D(
            filters=filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            activation=self.conv_activation,
        )(x)

        # 1# Pool
        x = keras.layers.MaxPooling3D(
            pool_size=(1, 2, 2),
            strides=(1, self.pooling_strides[1], self.pooling_strides[2]),
            padding=self.padding,
        )(x)

        filters = 2 * filters

        x = keras.layers.LayerNormalization()(x)

        # 2# Conv
        x = keras.layers.Conv3D(
            filters=filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            activation=self.conv_activation,
        )(x)

        # 2# Pool
        x = keras.layers.MaxPooling3D(
            strides=self.pooling_strides, padding=self.padding
        )(x)

        filters = 2 * filters

        x = keras.layers.LayerNormalization()(x)

        # 3# Conv
        x = keras.layers.Conv3D(
            filters=filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            activation=self.conv_activation,
        )(x)

        # 4# Conv
        x = keras.layers.Conv3D(
            filters=filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            activation=self.conv_activation,
        )(x)

        # 3# Pool
        x = keras.layers.MaxPooling3D(
            strides=self.pooling_strides, padding=self.padding
        )(x)

        filters = 2 * filters

        x = keras.layers.LayerNormalization()(x)

        # 5# Conv
        x = keras.layers.Conv3D(
            filters=filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            activation=self.conv_activation,
        )(x)

        # 6# Conv
        x = keras.layers.Conv3D(
            filters=filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            activation=self.conv_activation,
        )(x)

        # 4# Pool
        x = keras.layers.MaxPooling3D(
            strides=self.pooling_strides, padding=self.padding
        )(x)

        # 7# Conv
        x = keras.layers.Conv3D(
            filters=filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            activation=self.conv_activation,
        )(x)

        # 8# Conv
        x = keras.layers.Conv3D(
            filters=filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            activation=self.conv_activation,
        )(x)

        # 5# Pool
        x = keras.layers.MaxPooling3D(
            strides=self.pooling_strides, padding=self.padding
        )(x)

        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.GlobalAveragePooling3D()(x)
        x = keras.layers.Flatten()(x)

        x = keras.layers.Dense(4096)(x)
        x = keras.layers.Activation("swish")(x)
        x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.Dense(4096)(x)
        x = keras.layers.Activation("swish")(x)
        x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.Dense(14)(x)
        x = keras.layers.Softmax()(x)

        self._model = keras.Model(
            _input,
            x,
            name=self.__class__.__name__,
        )
        if self.continue_training:
            self.load_weights()
