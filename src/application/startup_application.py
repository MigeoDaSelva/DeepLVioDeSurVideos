from src.controller.setting_controller import SettingController
from src.infrastructure.simulation_scope_repository import SimulationScopeRepository
from src.infrastructure.data_attributes_repository import DataAttributesRepository
from src.infrastructure.attributes_repository import AttributesRepository
from src.infrastructure.approach_repository import ApproachRepository
from src.controller.widgets_controller import WidgetsController
from src.controller.approach_settings import ApproachSettings
from src.controller.scope_settings import ScopeSettings
from src.controller.data_settings import DataSettings


class StartupApplication:
    @classmethod
    def get_scope_manager(self) -> WidgetsController:
        return self._build_manager(ScopeSettings, SimulationScopeRepository)

    @classmethod
    def get_data_manager(self) -> WidgetsController:
        return self._build_manager(DataSettings, DataAttributesRepository)

    @classmethod
    def get_approach_manager(self) -> WidgetsController:
        return self._build_manager(ApproachSettings, ApproachRepository)

    @classmethod
    def _build_manager(
        self, setting_bar: SettingController, repository: AttributesRepository
    ) -> WidgetsController:
        repository = repository()
        repository.read_attributes()
        setting_bar = setting_bar(repository.attributes)
        setting_bar.build()
        setting_bar.show()
        return setting_bar
