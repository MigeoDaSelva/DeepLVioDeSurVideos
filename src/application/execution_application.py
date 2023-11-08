from src.controller.approach_settings import ApproachSettings
from src.engine.pipeline_factory import PipelineFactory
from src.controller.scope_settings import ScopeSettings
from src.controller.data_settings import DataSettings
from src.engine.pipeline import Pipeline


class ExecutionApplication:
    @classmethod
    def get_pipeline(
        self,
        simulation_scope_settings: ScopeSettings,
        data_settings: DataSettings,
        approach_settings: ApproachSettings,
    ) -> Pipeline:
        return PipelineFactory.creates(
            simulation_scope_settings,
            data_settings,
            approach_settings,
        )
