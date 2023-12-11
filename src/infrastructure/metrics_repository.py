from pandas import DataFrame, Series, concat, read_csv
from configs import settings


class MetricsRepository:
    @classmethod
    def write(self, file_name: str, metrics: dict) -> DataFrame:
        data_frame = DataFrame()
        item = dict()
        for metric in metrics:
            item["Metric"] = metric
            item.update(metrics[metric])
            data_frame = concat(
                [data_frame, Series(item).to_frame().T], ignore_index=True
            )

        data_frame.to_csv(f"{settings.CLASSIFICATION_METRICS}/{file_name}.csv")

    @classmethod
    def read(self, file_name: str) -> DataFrame:
        return read_csv(f"{settings.CLASSIFICATION_METRICS}/{file_name}.csv")

    @classmethod
    def update(self, file_name: str, metrics: dict) -> DataFrame:
        data_frame = self.read(file_name=file_name)
        item = dict()
        for metric in metrics:
            item["Metric"] = metric
            item.update(metrics[metric])
            data_frame = concat(
                [data_frame, Series(item).to_frame().T], ignore_index=True
            )

        data_frame.to_csv(f"{settings.CLASSIFICATION_METRICS}/{file_name}.csv")
