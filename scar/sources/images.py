"""Image sources: map frame timestamps to image files."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .poses import to_nanoseconds


@dataclass(frozen=True)
class FileListImageSource:
    """Images stored as individual files, one per timestamp [ns]."""

    timestamps_ns: np.ndarray
    paths: tuple[str, ...]
    root: Path | None = None

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_ns, dtype=np.int64).reshape(-1)
        if len(timestamps) != len(self.paths):
            raise ValueError("Timestamps and image paths differ in length")
        order = np.argsort(timestamps, kind="stable")
        object.__setattr__(self, "timestamps_ns", timestamps[order])
        object.__setattr__(self, "paths", tuple(str(self.paths[i]) for i in order))
        if self.root is not None:
            object.__setattr__(self, "root", Path(self.root))

    @classmethod
    def from_csv(cls, csv_path: str | Path, *, timestamp_column: str = "timestamp", path_column: str | None = "path",
        path_template: str | None = None, time_unit: str = "ns", root: str | Path | None = None) -> FileListImageSource:
        """Read image rows from a CSV file.

        The image path is taken from ``path_column`` or built with ``path_template`` from the row's
        columns (e.g. ``"{image_index}.png"``). Relative paths resolve against ``root`` (default:
        the CSV's folder).
        """
        csv_path = Path(csv_path)
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError(f"'{csv_path}' contains no images")
        # timestamps stay text until converted, so decimal seconds are converted exactly
        timestamps = to_nanoseconds(np.asarray([row[timestamp_column] for row in rows]), time_unit)
        if path_template is not None:
            paths = tuple(path_template.format(**row) for row in rows)
        else:
            paths = tuple(row[path_column] for row in rows)
        return cls(timestamps_ns=timestamps, paths=paths, root=Path(root) if root is not None else csv_path.parent)

    @classmethod
    def from_directory(cls, directory: str | Path, *, time_unit: str = "s", suffix: str = ".png") -> FileListImageSource:
        """Images named by their timestamp, e.g. ``1776855480.070000.png`` (seconds)."""
        directory = Path(directory)
        names = sorted(entry.name for entry in os.scandir(directory) if entry.name.endswith(suffix))
        if not names:
            raise ValueError(f"No '*{suffix}' images in '{directory}'")
        stems = np.asarray([name[: -len(suffix)] for name in names])
        return cls(timestamps_ns=to_nanoseconds(stems, time_unit), paths=tuple(names), root=directory)

    def select(self, *, start_ns: int | None = None, end_ns: int | None = None, step: int = 1,
        interval_ns: int | None = None, tolerance_ns: int | None = None) -> np.ndarray:
        """Indices of images in ``[start_ns, end_ns]``.

        Every ``step``-th one, or with ``interval_ns`` the image closest to each point of a regular
        time grid (skipped if farther than ``tolerance_ns``).
        """
        timestamps = self.timestamps_ns
        first = 0 if start_ns is None else int(np.searchsorted(timestamps, start_ns, side="left"))
        last = len(timestamps) if end_ns is None else int(np.searchsorted(timestamps, end_ns, side="right"))
        if interval_ns is None:
            return np.arange(first, last, max(1, int(step)))
        if first >= last:
            return np.zeros(0, dtype=int)

        grid = np.arange(timestamps[first], timestamps[last - 1] + 1, int(interval_ns))
        window = timestamps[first:last]
        position = np.clip(np.searchsorted(window, grid), 1, len(window) - 1) if len(window) > 1 else np.zeros(len(grid), int)
        before = window[position - 1] if len(window) > 1 else window[position]
        after = window[position]
        nearest = np.where(np.abs(grid - before) <= np.abs(after - grid), position - 1, position) if len(window) > 1 \
            else position
        keep = np.ones(len(grid), dtype=bool) if tolerance_ns is None else np.abs(window[nearest] - grid) <= tolerance_ns
        return np.unique(first + nearest[keep])

    def __len__(self) -> int:
        return len(self.paths)

    def resolve(self, path: str) -> Path:
        path = Path(path)
        return path if path.is_absolute() or self.root is None else self.root / path

    def read(self, path: str, grayscale: bool = True) -> np.ndarray:
        resolved = self.resolve(path)
        image = cv2.imread(str(resolved), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image '{resolved}'")
        return image
