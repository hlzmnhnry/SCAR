"""Conversion of geodetic navigation data into the SCAR world frame.

All transformations go through PROJ with ballpark transformations disabled: if PROJ lacks
the grid or operation needed, a conversion fails loudly instead of silently dropping, e.g.,
a 47 m geoid undulation.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from pyproj import CRS, Proj, Transformer
from pyproj.exceptions import ProjError

from scar.core.se3 import R_ENU_NED, rot_z

from .world_frame import HeightReference, WorldFrame


@lru_cache(maxsize=16)
def _horizontal_transformer(source_crs: str, target_crs: str) -> Transformer:
    source = CRS.from_user_input(source_crs)
    if not source.is_geographic:
        raise ValueError(f"Source CRS must be geographic (lat/lon), got {source.name!r}")
    return Transformer.from_crs(source.to_2d(), CRS.from_user_input(target_crs).to_2d(),
        always_xy=True, allow_ballpark=False, only_best=True)


@lru_cache(maxsize=16)
def _geoid_transformer(grid: str) -> Transformer:
    # resolve local files to absolute paths, keep bare names for PROJ's own grid search
    grid_path = Path(grid).expanduser()
    grid_ref = str(grid_path.resolve()) if grid_path.exists() else grid
    pipeline = (
        "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad "
        f"+step +proj=vgridshift +grids={grid_ref} +multiplier=1 "
        "+step +proj=unitconvert +xy_in=rad +xy_out=deg"
    )
    return Transformer.from_pipeline(pipeline)


def describe_horizontal_transformation(source_crs: str, world: WorldFrame) -> dict:
    """Name and stated accuracy [m] of the operation used from ``source_crs`` to the world CRS."""
    transformer = _horizontal_transformer(source_crs, world.crs)
    return {"description": transformer.description, "accuracy_m": transformer.accuracy}


def geoid_undulation(latitude: np.ndarray, longitude: np.ndarray, grid: str) -> np.ndarray:
    """Geoid undulation / height anomaly ``N`` [m] such that ``h_ellipsoidal = H + N``."""
    latitude, longitude = np.broadcast_arrays(np.asarray(latitude, dtype=float), np.asarray(longitude, dtype=float))
    try:
        transformer = _geoid_transformer(str(grid))
    except ProjError as error:
        raise ValueError(f"Geoid grid '{grid}' could not be loaded by PROJ") from error
    _, _, N = transformer.transform(longitude.ravel(), latitude.ravel(), np.zeros(latitude.size))
    N = np.asarray(N, dtype=float).reshape(latitude.shape)
    if not np.all(np.isfinite(N)):
        raise ValueError(f"Geoid grid '{grid}' does not cover all positions")
    return N


def to_ellipsoidal_height(latitude, longitude, height, reference: HeightReference,
    undulation: np.ndarray | None = None) -> np.ndarray:
    height = np.asarray(height, dtype=float)
    if reference.kind == "ellipsoidal":
        return height
    if reference.kind == "geoid":
        return height + geoid_undulation(latitude, longitude, reference.grid)
    if reference.kind == "undulation":
        if undulation is None:
            raise ValueError("Height reference 'undulation' needs per-sample undulation values")
        return height + np.asarray(undulation, dtype=float)
    raise ValueError(f"Cannot convert height reference '{reference.kind}' to ellipsoidal heights")


def convert_height(latitude, longitude, height, source: HeightReference, world: WorldFrame,
    undulation: np.ndarray | None = None) -> np.ndarray:
    """Convert heights into the vertical reference of the world frame."""
    if source.kind == "world" or source == world.height:
        return np.asarray(height, dtype=float)

    h_ellipsoidal = to_ellipsoidal_height(latitude, longitude, height, source, undulation)
    if world.height.kind == "ellipsoidal":
        return h_ellipsoidal
    return h_ellipsoidal - geoid_undulation(latitude, longitude, world.height.grid)


def geodetic_to_world(latitude, longitude, height, *, world: WorldFrame, height_reference: HeightReference,
    source_crs: str = "EPSG:4937", undulation: np.ndarray | None = None) -> np.ndarray:
    """Convert geodetic positions to local world coordinates (N, 3).

    ``source_crs`` is the geographic CRS of latitude/longitude (default ETRS89). Inputs are
    processed in float64; float32 latitudes quantize northing to ~0.4 m and must be avoided.
    """
    latitude = np.atleast_1d(np.asarray(latitude, dtype=np.float64))
    longitude = np.atleast_1d(np.asarray(longitude, dtype=np.float64))
    height = np.atleast_1d(np.asarray(height, dtype=np.float64))

    easting, northing = _horizontal_transformer(source_crs, world.crs).transform(longitude, latitude)
    up = convert_height(latitude, longitude, height, height_reference, world, undulation)

    points = np.stack([easting, northing, up], axis=-1)
    if not np.all(np.isfinite(points)):
        raise ValueError("Geodetic conversion produced non-finite coordinates")
    return world.to_local(points)


def meridian_convergence(latitude, longitude, world: WorldFrame) -> np.ndarray:
    """Angle [rad] of the rotation from local east-north-up axes to grid axes (about up).

    Uses pyproj's convention, e.g. about -1.37 deg at 7.22 E / 50.59 N in UTM zone 32N,
    where true north points 1.37 deg clockwise of grid north.
    """
    factors = Proj(world.pyproj_crs).get_factors(np.asarray(longitude, dtype=float), np.asarray(latitude, dtype=float))
    return np.radians(np.asarray(factors.meridian_convergence, dtype=float))


def world_from_navigation_rotation(latitude, longitude, world: WorldFrame, navigation_frame: str = "NED") -> np.ndarray:
    """Rotation(s) ``R_world_nav`` from a local navigation frame (NED or ENU, true north) to world axes."""
    latitude, longitude = np.broadcast_arrays(np.atleast_1d(np.asarray(latitude, dtype=float)),
        np.atleast_1d(np.asarray(longitude, dtype=float)))
    if world.apply_meridian_convergence:
        R_world_enu = rot_z(meridian_convergence(latitude, longitude, world))
    else:
        R_world_enu = np.broadcast_to(np.eye(3), (*latitude.shape, 3, 3)).copy()

    if navigation_frame == "ENU":
        return R_world_enu
    if navigation_frame == "NED":
        return R_world_enu @ R_ENU_NED
    raise ValueError(f"Unknown navigation frame {navigation_frame!r}, expected 'NED' or 'ENU'")


def orientation_to_world(R_nav_body: np.ndarray, latitude, longitude, world: WorldFrame,
    navigation_frame: str = "NED") -> np.ndarray:
    """Body orientations w.r.t. a local navigation frame -> ``R_world_body`` (N, 3, 3)."""
    R_world_nav = world_from_navigation_rotation(latitude, longitude, world, navigation_frame)
    return R_world_nav @ np.asarray(R_nav_body, dtype=float).reshape(R_world_nav.shape)
