from src.data_handler.strategies.class_names_finder import ClassNamesSequentialFinder
from sklearn.model_selection import StratifiedShuffleSplit
from functools import singledispatchmethod
from dataclasses import dataclass, field
from numpy import array, ndarray
from typing import List, Union
from random import shuffle
from pathlib import Path
import pickle
import configs.settings as settings
import os


@dataclass
class DataSplitter:
    dataset: List[Path]
    k_folds: int = field(default=10)
    n_iterations: int = field(default=30)
    test_size: Union[int, float, None] = field(default=None)
    train_size: Union[int, float, None] = field(default=None)
    validation_size: Union[int, float, None] = field(default=None)

    def __post_init__(self) -> None:
        if not (self.train_size or self.test_size):
            raise TypeError(
                "At least one of the attributes, test_size e train_size, must be initialized"
            )
        self._calculates_splits_size(self.train_size)

    def splits(self) -> None:
        shuffle(self.dataset)
        class_names = ClassNamesSequentialFinder().finds(file_paths=self.dataset)
        train_test_cross_validator = StratifiedShuffleSplit(
            n_splits=self.k_folds, train_size=self.train_size, test_size=self.test_size
        )
        train_validation_cross_validator = StratifiedShuffleSplit(
            n_splits=1,
            train_size=self.train_size,
            test_size=self.validation_size,
        )
        for i in range(self.n_iterations):
            train_test_indexes = train_test_cross_validator.split(
                X=self.dataset, y=class_names
            )
            for j, (j_train, j_test) in enumerate(train_test_indexes):
                x_train_val_set = array(self.dataset)[j_train]
                y_train_val_set = array(class_names)[j_train]
                train_validation_indexes = train_validation_cross_validator.split(
                    X=x_train_val_set, y=y_train_val_set
                )
                k_train, k_validation = next(train_validation_indexes)
                x_train_set = x_train_val_set[k_train]
                y_train_set = y_train_val_set[k_train]
                x_validation_set = x_train_val_set[k_validation]
                y_validation_set = y_train_val_set[k_validation]
                self._saves_dataset_fold(
                    dataset_fold=x_train_set, file_name=f"{i}_{j}_train"
                )
                self._saves_dataset_fold(
                    dataset_fold=x_validation_set, file_name=f"{i}_{j}_validation"
                )
                x_test_set = array(self.dataset)[j_test]
                y_test_set = array(class_names)[j_test]
                self._saves_dataset_fold(
                    dataset_fold=x_test_set, file_name=f"{i}_{j}_test"
                )

    def _saves_dataset_fold(self, dataset_fold: ndarray, file_name: str) -> None:
        cross_val_validation_path = settings.CROSS_VALIDATION_FILE_PATH
        dataset_fold = self._remove_relative_part(dataset_fold)
        if not os.path.exists(f"{cross_val_validation_path}/"):
            os.makedirs(f"{cross_val_validation_path}/")
        with open(f"{cross_val_validation_path}/{file_name}.pickle", "wb") as file:
            pickle.dump(dataset_fold, file, pickle.HIGHEST_PROTOCOL)

    def _remove_relative_part(self, dataset: ndarray) -> ndarray:
        for i, path in enumerate(dataset):
            dataset[i] = Path(f"/{path.parent.name}/{path.name}/")
        return dataset

    @singledispatchmethod
    def _calculates_splits_size(self, train_size) -> None:
        raise NotImplementedError(
            f"Method not yet implemented for this type: {type(train_size).__name__}"
        )

    @_calculates_splits_size.register
    def _(self, train_size: None) -> None:
        self.train_size = self._calculates_complement(self.test_size)

    @_calculates_splits_size.register
    def _(self, train_size: int) -> None:
        if not self.test_size:
            self.test_size = self._calculates_complement(train_size)

    @_calculates_splits_size.register
    def _(self, train_size: float) -> None:
        if not self.test_size:
            self.test_size = self._calculates_complement(train_size)

    @singledispatchmethod
    def _calculates_complement(self, split_size) -> Union[int, float]:
        raise NotImplementedError(
            f"Method not yet implemented for this type: {type(split_size).__name__}"
        )

    @_calculates_complement.register
    def _(self, split_size: int) -> Union[int, float]:
        return len(self.dataset) - split_size

    @_calculates_complement.register
    def _(self, split_size: float) -> Union[int, float]:
        return round(1 - split_size, 1)
