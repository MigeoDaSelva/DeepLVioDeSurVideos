from pandas import DataFrame, Series, concat, read_csv
from configs import settings


class MetricsRepository:
    @classmethod
    def write(self, file_name: str, metrics: dict) -> DataFrame:
        results = DataFrame()
        item = dict()
        for metric in metrics:
            item["Metric"] = metric
            item.update(metrics[metric])
            results = concat([results, Series(item).to_frame().T], ignore_index=True)

        results.to_csv(f"{settings.CLASSIFICATION_METRICS}/{file_name}.csv")

    @classmethod
    def read(self, file_name: str) -> DataFrame:
        return read_csv(f"{settings.CLASSIFICATION_METRICS}/{file_name}.csv")
