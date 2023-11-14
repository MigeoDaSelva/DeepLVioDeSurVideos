from src.data_handler.strategies.class_names_finder import UniqueClassNamesFinder
from src.data_handler.strategies.file_path_finder import RecursiveFilePathFinder
from src.engine.metrics_calculator import ClassificationMetricsCalculator
from src.engine.graphical_results_builder import GraphicalResultsBuilder
from src.data_handler.data_generator_factory import DataGeneratorFactory
from src.infrastructure.figure_repository import FigureRespository
from src.infrastructure.metrics_repository import MetricsRepository
from src.controller.widgets_controller import WidgetsController
from src.controller.result_controller import ResultController
from src.controller.graphical_result import GraphicalResult
from src.data_handler.dataset_factory import DatasetFactory
from src.domain.approach.abstract_approach import Approach
from src.controller.metrics_result import MetricsResult
from src.controller.scope_settings import ScopeSettings
from src.controller.data_settings import DataSettings
from dataclasses import dataclass, field
from datetime import datetime
from configs import settings
from pathlib import Path
from typing import List


@dataclass
class EvaluationApplication:
    approach: Approach
    data_settings: DataSettings
    simulation_scope_settings: ScopeSettings
    _labels: List[str] = field(init=False)

    def __post_init__(self) -> None:
        paths = RecursiveFilePathFinder(
            file_extensions=settings.SUPPORTED_VIDEO_EXTENSIONS
        ).finds(
            path=Path(f"{settings.DATASETS_PATH}/{settings.DATASET_NAME}"),
        )
        self._labels = UniqueClassNamesFinder().finds(paths)

    @property
    def labels(self) -> List[str]:
        return self._labels

    def evaluate(self) -> None:
        test_generator = DataGeneratorFactory.creates(
            path=Path(
                f"{settings.CROSS_VALIDATION_FILE_PATH}/{self.simulation_scope_settings.iterations[0]}_{self.simulation_scope_settings.folds[0]}_test.pickle"
            ),
            data_settings=self.data_settings,
        )
        test_generator.shuffle = False
        dataset_factory = DatasetFactory(
            batch_size=self.data_settings.batch_size,
            output_shape=(None, None, None, self.data_settings.n_channels),
        )
        test_ds = dataset_factory.creates(test_generator)
        self.approach.evaluate(test_ds)

    def calculate_classification_metrics(
        self,
    ) -> WidgetsController:
        file_name = f"{self.approach.model.name}-classification-metrics-({datetime.now().strftime('%d-%m-%Y-%H-%M')})"
        classification_metrics = (
            ClassificationMetricsCalculator.calculate_classification_metrics(
                self.approach,
                self.labels,
            )
        )
        MetricsRepository.write(file_name=file_name, metrics=classification_metrics)
        result = MetricsResult(MetricsRepository.read(file_name=file_name))
        result.build()
        result.show()
        return result

    def build_learning_curves(self) -> WidgetsController:
        file_name = f"{self.approach.model.name}-learning-curves-({datetime.now().strftime('%d-%m-%Y-%H-%M')})"
        GraphicalResultsBuilder.build_learning_curves(approach=self.approach)
        return self._save_and_show_figure(file_name=file_name)

    def build_confusion_matrix(
        self,
    ) -> WidgetsController:
        file_name = f"{self.approach.model.name}-test-confusion-matrix-({datetime.now().strftime('%d-%m-%Y-%H-%M')})"
        GraphicalResultsBuilder.build_confusion_matrix(
            approach=self.approach, labels=self.labels
        )
        return self._save_and_show_figure(file_name=file_name)

    def build_roc_curve(self) -> WidgetsController:
        file_name = f"{self.approach.model.name}-roc-curve-({datetime.now().strftime('%d-%m-%Y-%H-%M')})"
        GraphicalResultsBuilder.build_roc_curve(self.approach)
        return self._save_and_show_figure(file_name=file_name)

    def calculate_auc(self) -> float:
        return ClassificationMetricsCalculator.calculate_AUC(self.approach)

    def _save_and_show_figure(self, file_name: str) -> WidgetsController:
        FigureRespository.save(file_name=file_name)
        graphical_result = GraphicalResult(FigureRespository.load(file_name=file_name))
        graphical_result.build()
        graphical_result.show()
        return graphical_result
