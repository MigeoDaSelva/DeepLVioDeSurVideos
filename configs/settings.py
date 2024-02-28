from os.path import join, dirname
from dotenv import load_dotenv
from pathlib import Path
import ast
import os

env_file_path: str = join(dirname(__file__), "env_file.env")
load_dotenv(env_file_path)

# Global
PROJECT_ROOT_PATH: Path = Path(dirname(os.path.realpath(__file__))).parent

COLAB_RELEASE_TAG: str = os.popen("echo $COLAB_RELEASE_TAG").read().rstrip()

COLAB_ENV: bool = any(COLAB_RELEASE_TAG)


def _decides_source(env_variable: str) -> Path:
    return (
        Path(
            f"{os.environ.get('GOOGLE_DRIVE_PROJECT_PATH')}{os.environ.get(env_variable)}"
        )
        if COLAB_ENV
        else Path(f"{PROJECT_ROOT_PATH}{os.environ.get(env_variable)}")
    )


_file_ext_string = os.environ.get("SUPPORTED_VIDEO_EXTENSIONS")
SUPPORTED_VIDEO_EXTENSIONS: list = (
    ast.literal_eval(_file_ext_string) if _file_ext_string else []
)
DATASET_NAME: str = str(os.environ.get("DATASET_NAME"))
MOVINET_VERSION: str = str(os.environ.get("MOVINET_VERSION"))
MOVINET_URL: str = str(os.environ.get("MOVINET_URL")).replace(
    "<VERSION>", MOVINET_VERSION
)

# Path definitions
IMAGES_PATH: Path = _decides_source("IMAGES_PATH")
DATASETS_PATH: Path = _decides_source("DATASETS_PATH")
MODEL_CHECKPOINT_PATH: Path = _decides_source("MODEL_CHECKPOINT_PATH")
CROSS_VALIDATION_PATH: Path = _decides_source("CROSS_VALIDATION_PATH")
CONFUSION_MATRIX_PATH: Path = _decides_source("CONFUSION_MATRIX_PATH")
CLASSIFICATION_METRICS: Path = _decides_source("CLASSIFICATION_METRICS")
MODEL_TRAINING_HISTORY_PATH: Path = _decides_source("MODEL_TRAINING_HISTORY_PATH")


EXISTING_CROSS_VALIDATION_FILES: list = list(
    filter(
        lambda path: not path.name.startswith("."),
        CROSS_VALIDATION_PATH.glob("*.*"),
    )
)

EXISTING_MODEL_CHECKPOINT: list = list(
    filter(
        lambda path: path.is_dir() and not path.name.startswith("lite_models"),
        MODEL_CHECKPOINT_PATH.iterdir(),
    )
)

EXISTING_CONFUSION_MATRIX_FILES: list = list(
    filter(
        lambda path: not path.name.startswith("."),
        CONFUSION_MATRIX_PATH.glob("*.*"),
    )
)

EXISTING_CLASSIFICATION_METRICS: list = list(
    filter(
        lambda path: "average" not in path.name and not path.name.startswith("."),
        CLASSIFICATION_METRICS.glob("*.*"),
    )
)

# Default setting files
DATA_SETTINGS_FILE: Path = Path(
    f"{PROJECT_ROOT_PATH}{os.environ.get('DATA_SETTINGS_FILE')}"
)
APPROACH_SETTINGS_FILE: Path = Path(
    f"{PROJECT_ROOT_PATH}{os.environ.get('APPROACH_SETTINGS_FILE')}"
)
SCOPE_SETTINGS_FILE: Path = Path(
    f"{PROJECT_ROOT_PATH}{os.environ.get('SCOPE_SETTINGS_FILE')}"
)
