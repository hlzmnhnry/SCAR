"""Definition of the metric world frame SCAR optimizes in.

The world frame is a projected CRS (e.g. ETRS89 / UTM 32N, EPSG:25832) whose axes are
grid east, grid north and up, with heights in an explicit vertical reference. It must
match the reference data (orthophotos and DSM), because landmarks are read from them.

Coordinates stored by SCAR are relative to ``origin`` to keep numbers small.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pyproj import CRS

HeightKind = Literal["ellipsoidal", "geoid", "undulation", "world"]


@dataclass(frozen=True)
class HeightReference:
    """Vertical reference of a height value.

    - ``ellipsoidal``: height above the ellipsoid of the horizontal datum.
    - ``geoid``: height above a geoid/quasigeoid given as a PROJ-readable grid, i.e.
      ``h_ellipsoidal = H + N(lat, lon)``. ``grid`` is a file path or a PROJ grid name.
    - ``undulation``: height above a geoid whose undulation ``N`` is given per sample
      (e.g. the ``undulation`` column of SBG INS logs), ``h_ellipsoidal = H + N``. Inputs only.
    - ``world``: heights are already in the vertical reference of the world frame and are
      used unchanged. Inputs only; this is how the legacy pipeline treated INS heights.
    """

    kind: HeightKind
    grid: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("ellipsoidal", "geoid", "undulation", "world"):
            raise ValueError(f"Unknown height reference kind {self.kind!r}")
        if self.kind == "geoid" and not self.grid:
            raise ValueError("A geoid height reference needs a grid")
        if self.kind != "geoid" and self.grid is not None:
            raise ValueError(f"Height reference '{self.kind}' does not take a grid")

    @classmethod
    def ellipsoidal(cls) -> HeightReference:
        return cls("ellipsoidal")

    @classmethod
    def geoid(cls, grid: str, name: str | None = None) -> HeightReference:
        return cls("geoid", grid=str(grid), name=name)

    @classmethod
    def undulation(cls, name: str | None = None) -> HeightReference:
        return cls("undulation", name=name)

    @classmethod
    def world(cls) -> HeightReference:
        return cls("world")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "grid": self.grid, "name": self.name}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeightReference:
        return cls(kind=data["kind"], grid=data.get("grid"), name=data.get("name"))


@dataclass(frozen=True)
class WorldFrame:
    """Projected world frame: grid axes, explicit heights and a local origin.

    ``apply_meridian_convergence`` controls whether orientations given w.r.t. true north are
    rotated into grid axes. It exists to reproduce the legacy pipeline, which did not do this.
    """

    crs: str
    height: HeightReference
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    apply_meridian_convergence: bool = True
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        crs = CRS.from_user_input(self.crs)
        if not crs.is_projected:
            raise ValueError(f"World CRS must be projected, got {crs.name!r}")
        if self.height.kind not in ("ellipsoidal", "geoid"):
            raise ValueError("The world frame needs an explicit vertical reference (ellipsoidal or geoid)")
        origin = np.asarray(self.origin, dtype=float)
        if origin.shape != (3,):
            raise ValueError("origin must have 3 entries")
        object.__setattr__(self, "origin", tuple(float(v) for v in origin))

    @property
    def pyproj_crs(self) -> CRS:
        return CRS.from_user_input(self.crs)

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """Absolute projected coordinates (..., 3) -> coordinates relative to the origin."""
        return np.asarray(points, dtype=float) - np.asarray(self.origin)

    def to_absolute(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float) + np.asarray(self.origin)

    def with_origin(self, origin) -> WorldFrame:
        return WorldFrame(crs=self.crs, height=self.height, origin=tuple(origin),
            apply_meridian_convergence=self.apply_meridian_convergence, metadata=dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "crs": self.crs,
            "height": self.height.to_dict(),
            "origin": list(self.origin),
            "apply_meridian_convergence": self.apply_meridian_convergence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorldFrame:
        return cls(
            crs=data["crs"],
            height=HeightReference.from_dict(data["height"]),
            origin=tuple(data.get("origin", (0.0, 0.0, 0.0))),
            apply_meridian_convergence=data.get("apply_meridian_convergence", True),
            metadata=data.get("metadata", {}),
        )
