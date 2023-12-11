from configs import settings
from urllib import request
from pathlib import Path
import tarfile
import os


class PreTrainedModelsDownloader:
    @classmethod
    def download_movinet(self) -> None:
        file_name = Path(settings.MOVINET_URL).name
        file_path = f"{settings.PRE_TRAINED_MODELS_PATH}/{file_name}"
        request.urlretrieve(
            settings.MOVINET_URL,
            file_path,
        )

        with tarfile.open(file_path) as tar:
            tar.extractall(settings.PRE_TRAINED_MODELS_PATH)
            tar.close()
        os.system(f"rm {file_path}")
