from src.domain.approach.abstract_approach import Approach
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from typing import List
import tensorflow as tf
import seaborn as sns
import numpy as np


class GraphicalResultsBuilder:
    @classmethod
    def build_confusion_matrix(self, approach: Approach, labels: List[str]) -> None:
        cm = tf.math.confusion_matrix(approach.actual, approach.predicted)
        ax = sns.heatmap(cm, annot=True, fmt="g")
        sns.set(rc={"figure.figsize": (35, 35)})
        sns.set(font_scale=5)
        ax.set_xlabel("Ação prevista")
        ax.set_ylabel("Ação real")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.subplots_adjust(bottom=0.2, top=0.95, left=0.2, right=1)
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)

    @classmethod
    def build_roc_curve(self, approach: Approach) -> None:
        display = RocCurveDisplay.from_predictions(approach.actual, approach.predicted)
        display.figure_.set_size_inches(10, 10)
        plt.rcParams.update({"font.size": 20})

    @classmethod
    def build_learning_curves(self, approach: Approach) -> None:
        fig, (ax1, ax2) = plt.subplots(2)

        fig.set_size_inches(10, 15)

        # Plot loss
        ax1.set_title("Perda")
        ax1.plot(approach.history.history["loss"], label="train")
        ax1.plot(approach.history.history["val_loss"], label="test")
        ax1.set_ylabel("Perda")

        # Determine upper bound of y-axis
        max_loss = (
            max(approach.history.history["loss"])
            if max(approach.history.history["loss"])
            > max(approach.history.history["val_loss"])
            else max(approach.history.history["val_loss"])
        )
        min_loss = (
            min(approach.history.history["loss"])
            if min(approach.history.history["loss"])
            < min(approach.history.history["val_loss"])
            else min(approach.history.history["val_loss"])
        )

        ax1.set_ylim([min_loss, np.ceil(max_loss)])
        ax1.set_xlabel("Época")
        ax1.legend(["Treino", "Validação"])

        # Plot Accuracy
        ax2.set_title("Acurácia")
        ax2.plot(approach.history.history["accuracy"], label="train")
        ax2.plot(approach.history.history["val_accuracy"], label="test")
        ax2.set_ylabel("Acurácia")
        max_accuracy = (
            max(approach.history.history["accuracy"])
            if max(approach.history.history["accuracy"])
            > max(approach.history.history["val_accuracy"])
            else max(approach.history.history["val_accuracy"])
        )
        min_accuracy = (
            min(approach.history.history["accuracy"])
            if min(approach.history.history["accuracy"])
            < min(approach.history.history["val_accuracy"])
            else min(approach.history.history["val_accuracy"])
        )
        ax2.set_ylim([min_accuracy, np.ceil(max_accuracy)])
        ax2.set_xlabel("Época")
        ax2.legend(["Treino", "Validação"])
