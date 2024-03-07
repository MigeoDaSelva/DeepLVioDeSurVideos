from pandas import DataFrame, Series, concat, read_csv
from configs import settings
from pathlib import Path


class MetricsRepository:
    @classmethod
    def write(self, file_name: str, metrics: dict) -> None:
        for metric in metrics:
            item = {}
            current_file_name = file_name.replace("metric", metric)
            current_file_path = Path(
                f"{settings.CLASSIFICATION_METRICS}/{current_file_name}.csv"
            )
            if current_file_path.exists():
                data_frame = self.read(current_file_path)
            else:
                data_frame = DataFrame()

            item.update(metrics[metric])
            data_frame = concat(
                [data_frame, Series(item).to_frame().T], ignore_index=True
            )
            data_frame.to_csv(current_file_path, index=False)

    @classmethod
    def read(self, file_path: Path) -> DataFrame:
        return read_csv(file_path)

    @classmethod
    def update(self, file_name: str, metrics: dict) -> DataFrame:
        for metric in metrics:
            item = {}
            current_file_name = file_name.replace("metric", metric)
            current_file_path = Path(
                f"{settings.CLASSIFICATION_METRICS}/{current_file_name}.csv"
            )
            data_frame = self.read(current_file_path)
            item.update(metrics[metric])
            data_frame = concat(
                [data_frame, Series(item).to_frame().T], ignore_index=True
            )

            data_frame.to_csv(current_file_path)
