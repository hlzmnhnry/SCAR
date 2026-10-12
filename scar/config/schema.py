"""Configuration of a SCAR run (``scar run config.yaml``).

A run describes the sequence (camera, initial extrinsics, images, navigation poses), the world
frame, where reference data comes from, how anchors are collected and which optimization preset
is used. Relative paths are resolved against the directory of the configuration file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from scar.collect.collector import CollectConfig
from scar.datasets.generic_csv import PoseCsvSpec
from scar.references.georay import GeorayReferenceConfig

from .loader import ConfigError, from_dict


@dataclass(frozen=True)
class ImagesConfig:
    """Where the images are and when they were taken.

    ``type``:
      - ``csv``: rows of ``csv`` with ``timestamp_column`` and ``path_column`` (or ``path_template``
        built from the row, e.g. ``"frame_images/{image_index}.png"``); paths relative to ``root``
        (default: the CSV's folder)
      - ``directory``: files in ``directory`` named by their timestamp (e.g. reconstructed event
        frames ``1776855480.070000.png``)
      - ``ltc``: an LTC campaign folder (``camera.csv``) with images in ``root``
    """

    type: str = "csv"
    csv: str | None = None
    directory: str | None = None
    root: str | None = None
    timestamp_column: str = "timestamp"
    path_column: str | None = "path"
    path_template: str | None = None
    time_unit: str = "s"
    suffix: str = ".png"

    def __post_init__(self) -> None:
        if self.type not in ("csv", "directory", "ltc"):
            raise ValueError("images.type must be 'csv', 'directory' or 'ltc'")


@dataclass(frozen=True)
class PosesConfig:
    """Navigation (INS/GNSS) poses of the body frame.

    ``type``: ``csv`` (described by ``format``, see :class:`~scar.datasets.generic_csv.PoseCsvSpec`),
    ``ltc`` (an LTC ``ins.csv``, converted with :mod:`scar.geodesy`) or ``ltc_legacy`` (exactly as
    in the CVPRW 2026 pipeline). Pose priors are evaluated at ``image time + time_offset_s``.
    """

    type: str = "csv"
    csv: str | None = None
    format: PoseCsvSpec = field(default_factory=PoseCsvSpec)
    interpolation: str = "linear"
    max_gap_s: float = 0.05
    time_offset_s: float = 0.0
    ltc_height: str = "undulation"

    def __post_init__(self) -> None:
        if self.type not in ("csv", "ltc", "ltc_legacy"):
            raise ValueError("poses.type must be 'csv', 'ltc' or 'ltc_legacy'")


@dataclass(frozen=True)
class SelectionConfig:
    """Which images to use.

    ``[start, end]`` (seconds, as in the image timestamps), then every ``step``-th image, or the
    image nearest to every ``interval_s`` (within ``tolerance_s``).
    """

    start: str | None = None
    end: str | None = None
    step: int = 1
    interval_s: float | None = None
    tolerance_s: float | None = None


@dataclass(frozen=True)
class SequenceConfig:
    camera: str
    extrinsics: str
    images: ImagesConfig
    poses: PosesConfig
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    camera_key: str | None = None
    mask: str | None = None
    hot_pixels: str | None = None
    hot_pixel_radius: int = 1


@dataclass(frozen=True)
class WorldConfig:
    """Projected CRS and vertical reference of the reference data (and thus of the calibration)."""

    crs: str = "EPSG:25832"
    height: str = "ellipsoidal"
    geoid_grid: str | None = None
    height_name: str | None = None
    meridian_convergence: bool = True


@dataclass(frozen=True)
class CollectSection:
    matcher: str = "superpoint-lightglue"
    device: str = "auto"
    settings: CollectConfig = field(default_factory=CollectConfig)


@dataclass(frozen=True)
class OptimizeConfig:
    preset: str = "default"


@dataclass(frozen=True)
class ViewerConfig:
    """rerun viewer: live (``spawn``) and/or recorded to ``save`` (.rrd)."""

    enabled: bool = False
    spawn: bool = True
    save: str | None = None


@dataclass(frozen=True)
class RunConfig:
    output: str
    sequence: SequenceConfig
    world: WorldConfig = field(default_factory=WorldConfig)
    reference: GeorayReferenceConfig = field(default_factory=GeorayReferenceConfig)
    collect: CollectSection = field(default_factory=CollectSection)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    name: str = "scar"
    base_directory: str = "."

    def path(self, value: str | None) -> Path | None:
        """Resolve a path of the configuration relative to its file."""
        if value is None:
            return None
        path = Path(value).expanduser()
        return (path if path.is_absolute() else Path(self.base_directory) / path).resolve()


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping")
    data.setdefault("base_directory", str(path.resolve().parent))
    return from_dict(RunConfig, data)
