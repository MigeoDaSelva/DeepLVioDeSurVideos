import configs.settings as settings
from pathlib import Path
import tensorflow as tf
import pickle
import os


class ConfusionMatrixRepository:

    @classmethod
    def save(self, confusion_matrix: tf.Tensor, file_name: str) -> None:
        confusion_matrix_path = settings.CONFUSION_MATRIX_PATH
        if not os.path.exists(f"{confusion_matrix_path}/"):
            os.makedirs(f"{confusion_matrix_path}/")
        with open(f"{confusion_matrix_path}/{file_name}.pickle", "wb") as file:
            pickle.dump(confusion_matrix, file, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(self, path: Path) -> tf.Tensor:
        with open(path, "rb") as file:
            data = pickle.load(file)
        return data
