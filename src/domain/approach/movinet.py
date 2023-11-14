from src.domain.approach.pre_trained_approach import PreTrainedApproach
from src.infrastructure.pre_trained_models_downloader import (
    PreTrainedModelsDownloader as Downloader,
)
from official.projects.movinet.modeling import movinet_model
from official.projects.movinet.modeling import movinet
from configs import settings
from pathlib import Path
import tensorflow as tf


class Movinet(PreTrainedApproach):
    def build(self) -> None:
        if not Path(
            f"{settings.PRE_TRAINED_MODELS_PATH}/{settings.MOVINET_VERSION}"
        ).exists():
            Downloader.download_movinet()
        backbone = movinet.Movinet(model_id=settings.MOVINET_VERSION.split("_")[1])

        model = movinet_model.MovinetClassifier(
            backbone=backbone,
            num_classes=600,
        )

        model.build(self.input_shape)

        checkpoint_path = tf.train.latest_checkpoint(
            settings.PRE_TRAINED_CHECKPOINT_PATH
        )
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(checkpoint_path)
        status.expect_partial()
        status.assert_existing_objects_matched()
        self._model = movinet_model.MovinetClassifier(
            backbone=backbone,
            num_classes=10,
            name=self.__class__.__name__,
        )
        self.model.build(self.input_shape)

        if not self.unfreezing:
            for layer in self.model.layers[:-1]:
                layer.trainable = False
            self.model.layers[-1].trainable = True

    def build_only_base(self) -> None:
        backbone = movinet.Movinet(model_id=settings.MOVINET_VERSION.split("_")[1])

        backbone.trainable = self.unfreezing

        self._model = movinet_model.MovinetClassifier(
            backbone=backbone,
            num_classes=10,
            name=self.__class__.__name__,
        )
