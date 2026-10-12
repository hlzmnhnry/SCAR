"""Interface between SCAR and providers of georeferenced reference data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from scar.core.camera import CameraModel


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    mapped = points @ matrix[:, :2].T + matrix[:, 2]
    return mapped[:, :2] / mapped[:, 2:3]


@dataclass(frozen=True)
class ReferenceView:
    """Orthophoto and DSM around one camera footprint, on a common pixel grid.

    ``pixel_to_world`` (3x3, homogeneous) maps continuous pixel coordinates to local world
    coordinates (x, y) of the sequence's :class:`~scar.geodesy.WorldFrame`. Pixel (i, j) covers
    ``[i, i+1) x [j, j+1)``. DSM heights are in the world frame's vertical reference (relative
    to its origin); missing heights are NaN.
    """

    image: np.ndarray
    dsm: np.ndarray
    pixel_to_world: np.ndarray
    footprint: np.ndarray | None = None  # (4, 2) local world coordinates
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image.shape[:2] != self.dsm.shape[:2]:
            raise ValueError(f"Image {self.image.shape[:2]} and DSM {self.dsm.shape[:2]} must share the pixel grid")
        matrix = np.asarray(self.pixel_to_world, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError("pixel_to_world must be a 3x3 matrix")
        object.__setattr__(self, "pixel_to_world", matrix)

    @property
    def world_to_pixel(self) -> np.ndarray:
        return np.linalg.inv(self.pixel_to_world)

    def pixels_to_world(self, pixels: np.ndarray) -> np.ndarray:
        return _apply(self.pixel_to_world, pixels)

    def world_to_pixels(self, points: np.ndarray) -> np.ndarray:
        return _apply(self.world_to_pixel, points)

    @property
    def ground_sampling_distance(self) -> float:
        """Mean pixel size on the ground [m]."""
        return float(np.sqrt(abs(np.linalg.det(self.pixel_to_world[:2, :2]))))

    def heights(self, pixels: np.ndarray, mode: str = "nearest") -> np.ndarray:
        """DSM heights at continuous pixel positions (N, 2); NaN outside or where missing.

        ``nearest`` takes the pixel containing the position (as the legacy pipeline did);
        ``bilinear`` interpolates between pixel centres.
        """
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        height, width = self.dsm.shape[:2]
        dsm = self.dsm.astype(float)
        result = np.full(len(pixels), np.nan)

        inside = (pixels[:, 0] >= 0) & (pixels[:, 1] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] < height)
        u, v = pixels[inside, 0], pixels[inside, 1]
        if mode == "nearest":
            result[inside] = dsm[np.floor(v).astype(int), np.floor(u).astype(int)]
        elif mode == "bilinear":
            cu = np.clip(u - 0.5, 0.0, width - 1.0)
            cv = np.clip(v - 0.5, 0.0, height - 1.0)
            u0, v0 = np.floor(cu).astype(int), np.floor(cv).astype(int)
            u1, v1 = np.minimum(u0 + 1, width - 1), np.minimum(v0 + 1, height - 1)
            du, dv = cu - u0, cv - v0
            top = (1 - du) * dsm[v0, u0] + du * dsm[v0, u1]
            bottom = (1 - du) * dsm[v1, u0] + du * dsm[v1, u1]
            result[inside] = (1 - dv) * top + dv * bottom
        else:
            raise ValueError(f"Unknown height lookup {mode!r}")
        return result


class ReferenceProvider(Protocol):
    """Renders reference data for a camera pose (local world coordinates of the sequence)."""

    def render(self, T_world_camera: np.ndarray, camera: CameraModel, *, pixels_per_meter: float | None = None,
        timestamp_ns: int | None = None) -> ReferenceView:
        ...
