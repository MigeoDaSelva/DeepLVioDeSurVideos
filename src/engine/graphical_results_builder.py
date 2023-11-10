from src.domain.approach.abstract_approach import Approach
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from typing import List
import tensorflow as tf
import seaborn as sns


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
