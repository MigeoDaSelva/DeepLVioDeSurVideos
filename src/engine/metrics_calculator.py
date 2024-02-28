from src.infrastructure.metrics_repository import MetricsRepository
from src.domain.approach.abstract_approach import Approach
from src.controller.data_settings import DataSettings
from sklearn.metrics import roc_auc_score
from configs import settings
from typing import List
import tensorflow as tf
import numpy as np


class ClassificationMetricsCalculator:
    @classmethod
    def calculate_classification_metrics(
        self, approach: Approach, labels: List[str], data_settings: DataSettings
    ) -> dict:

        cm = tf.math.confusion_matrix(approach.actual, approach.predicted)
        tp = np.diag(cm)  # Diagonal represents true positives
        precision = {}
        recall = {}
        f1_score = {}
        for i, label in enumerate(labels):
            col = cm[:, i]
            fp = (
                np.sum(col) - tp[i]
            )  # Sum of column minus true positive is false negative

            row = cm[i, :]
            fn = (
                np.sum(row) - tp[i]
            )  # Sum of row minus true positive, is false negative

            precision_value = tp[i] / (tp[i] + fp)

            recall_value = tp[i] / (tp[i] + fn)

            precision[label] = precision_value  # Precision

            recall[label] = recall_value  # Recall

            f1_score[label] = 2 * (
                (precision_value * recall_value) / (precision_value + recall_value)
            )  # f1-score

        loss_accuracy_throughput = {
            "loss": approach.model.get_metrics_result()["loss"].numpy(),
            "accuracy": approach.model.get_metrics_result()["accuracy"].numpy(),
            "throughput": self.calculate_throughput(
                approach=approach, data_settings=data_settings
            ),
        }

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "loss_accuracy_throughput": loss_accuracy_throughput,
        }

    @classmethod
    def calculate_average_classification_metrics(self) -> dict:
        average_metrics = {}
        for file_path in settings.EXISTING_CLASSIFICATION_METRICS:
            file = MetricsRepository.read(file_path)
            averages = file.mean().to_dict()
            new_file_name = file_path.stem
            average_metrics[new_file_name] = averages

        return average_metrics

    @classmethod
    def calculate_AUC(self, approach: Approach) -> float:
        return roc_auc_score(approach.actual, approach.predicted)

    @classmethod
    def calculate_throughput(
        self, approach: Approach, data_settings: DataSettings
    ) -> float:
        return (
            (approach.actual.shape[0] / data_settings.batch_size)
            * data_settings.batch_size
        ) / approach.prediction_time
