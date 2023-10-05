from os.path import join, dirname
from dotenv import load_dotenv
from pathlib import Path
import ast
import os

env_file_path = join(dirname(__file__), "env_file.env")
load_dotenv(env_file_path)

# Global
PROJECT_ROOT_PATH = Path(dirname(os.path.realpath(__file__)))

_file_ext_string = os.environ.get("FILE_EXTENSIONS")
FILE_EXTENSIONS = ast.literal_eval(_file_ext_string) if _file_ext_string else []


# Dataset
DATASET_NAME = Path(f"{os.environ.get('DATASET_NAME')}")
CROSS_VALIDATION_PATH = Path(
    f"{PROJECT_ROOT_PATH}{os.environ.get('CROSS_VALIDATION_PATH')}"
)
DATASETS_PATH = Path(f"{PROJECT_ROOT_PATH}{os.environ.get('DATASETS_PATH')}")

# Images
IMAGES_PATH = Path(f"{PROJECT_ROOT_PATH}{os.environ.get('IMAGES_PATH')}")
