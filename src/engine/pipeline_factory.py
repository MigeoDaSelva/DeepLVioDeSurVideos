from src.data_handler.data_generator_factory import DataGeneratorFactory
from src.infrastructure.approach_repository import ApproachRepository
from src.controller.approach_settings import ApproachSettings
from src.data_handler.dataset_factory import DatasetFactory
from src.engine.approach_factory import ApproachFactory
from src.controller.scope_settings import ScopeSettings
from src.controller.data_settings import DataSettings
from src.engine.pipeline import Pipeline
from configs import settings
from pathlib import Path


class PipelineFactory:
    @classmethod
    def creates(
        self,
        simulation_scope_settings: ScopeSettings,
        data_settings: DataSettings,
        approach_settings: ApproachSettings,
    ) -> Pipeline:
        train_generator = DataGeneratorFactory.creates(
            path=Path(
                f"{settings.CROSS_VALIDATION_FILE_PATH}/{simulation_scope_settings.iterations[0]}_{simulation_scope_settings.folds[0]}_train.pickle"
            ),
            data_settings=data_settings,
        )

        validation_generator = DataGeneratorFactory.creates(
            path=Path(
                f"{settings.CROSS_VALIDATION_FILE_PATH}/{simulation_scope_settings.iterations[0]}_{simulation_scope_settings.folds[0]}_validation.pickle"
            ),
            data_settings=data_settings,
        )
        approach = ApproachFactory.creates(
            data_settings=data_settings, approach_settings=approach_settings
        )

        if approach_settings.load_latest_model:
            approach.unfreezing = approach_settings.unfreezing
            approach.build_only_base()
            approach = ApproachRepository.loads_latest_model_checkpoint(
                approach=approach
            )

        dataset_factory = DatasetFactory(
            batch_size=data_settings.batch_size,
            output_shape=(None, None, None, data_settings.n_channels),
        )

        return Pipeline(
            build_model=not approach_settings.load_latest_model,
            approach=approach,
            train_dataset=train_generator,
            validation_dataset=validation_generator,
            dataset_factory=dataset_factory,
        )
