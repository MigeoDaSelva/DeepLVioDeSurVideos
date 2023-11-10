from src.domain.approach.abstract_approach import Approach
from sklearn.metrics import roc_auc_score
from typing import List
import tensorflow as tf
import numpy as np


class ClassificationMetricsCalculator:
    @classmethod
    def calculate_classification_metrics(
        self, approach: Approach, labels: List[str]
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

        return {"precision": precision, "recall": recall, "f1_score": f1_score}

    @classmethod
    def calculate_AUC(self) -> float:
        return None
