from src.infrastructure.pre_trained_models_downloader import (
    PreTrainedModelsDownloader as Downloader,
)
from official.projects.movinet.modeling import movinet_model
from src.domain.approach.abstract_approach import Approach
from official.projects.movinet.modeling import movinet
from configs import settings
import tensorflow as tf


class Movinet(Approach):
    def build(self) -> None:
        tf.keras.backend.clear_session()

        backbone = movinet.Movinet(model_id=settings.MOVINET_VERSION.split("_")[1])

        backbone.trainable = self.unfreezing

        model = movinet_model.MovinetClassifier(
            backbone=backbone,
            num_classes=600,
        )
        
        model.build([None, None, None, None, self.input_shape[4]])

        if not (settings.MODEL_CHECKPOINT_PATH / settings.MOVINET_VERSION).exists():
            Downloader.download_movinet()

        checkpoint_path = tf.train.latest_checkpoint(
            settings.MODEL_CHECKPOINT_PATH / settings.MOVINET_VERSION
        )

        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(checkpoint_path)
        status.expect_partial()
        status.assert_existing_objects_matched()

        self._model = movinet_model.MovinetClassifier(
            backbone=backbone,
            num_classes=14,
            dropout_rate=self.dropout,
            name=self.__class__.__name__,
        )

        self.model.build(self.input_shape)

        if self.continue_training:
            self.load_weights()

    def build_only_base(self) -> None:
        backbone = movinet.Movinet(model_id=settings.MOVINET_VERSION.split("_")[1])

        backbone.trainable = self.unfreezing

        self._model = movinet_model.MovinetClassifier(
            backbone=backbone,
            num_classes=14,
            name=self.__class__.__name__,
        )
