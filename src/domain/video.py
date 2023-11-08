from dataclasses import dataclass, field
from numpy import ndarray
from pathlib import Path


@dataclass
class Video:
    _name: str = field(init=False)
    _label: str = field(init=False)
    _path: Path = field(init=False)
    _frames: ndarray = field(init=False)

    @property
    def length(self) -> int:
        return len(self.frames)

    @property
    def frames(self) -> ndarray:
        return self._frames

    @frames.setter
    def frames(self, frames: ndarray) -> None:
        self._frames = frames

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, path: Path) -> None:
        self._path = path

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, label: str) -> None:
        self._label = label

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Length: {self.length}\n"
            f"Label: {self.label}\n"
            f"Path: {self.path}\n"
            f"Array Shape: {self.frames.shape}"
        )
