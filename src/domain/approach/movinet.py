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
        Downloader.download_movinet()
        backbone = movinet.Movinet()
        backbone.trainable = False

        model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
        model.build([None, None, None, None, 3])

        checkpoint_path = tf.train.latest_checkpoint(
            settings.PRE_TRAINED_CHECKPOINT_PATH
        )
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint_path = checkpoint.save(checkpoint_path)
        checkpoint.restore(checkpoint_path)
        self._model = movinet_model.MovinetClassifier(
            backbone=backbone, num_classes=14, name="Movinet"
        )
