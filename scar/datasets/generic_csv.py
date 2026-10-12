"""Navigation poses from any CSV file, described by column names and conventions."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scar.core.se3 import euler_zyx_to_matrix, make_transform, quat_xyzw_to_matrix
from scar.geodesy import (
    HeightReference,
    WorldFrame,
    convert_height,
    geodetic_to_world,
    world_from_navigation_rotation,
)
from scar.sources.poses import PoseTable, ins_accuracy_covariance, to_nanoseconds


@dataclass(frozen=True)
class PoseCsvSpec:
    """How to read body poses from a CSV file.

    position
        ``geodetic``: ``columns`` = latitude, longitude, height in ``crs`` (a geographic CRS, e.g.
        EPSG:4979 for WGS84, EPSG:4937 for ETRS89). ``projected``: ``columns`` = x, y, height in
        the projected ``crs`` (e.g. UTM coordinates exported by the INS software), converted to the
        world CRS if it differs.
    height
        Vertical reference of the height column (``ellipsoidal``, ``geoid`` + ``geoid_grid``,
        ``undulation`` + ``undulation_column``, or ``world`` = already in the world's reference).
    orientation
        ``quaternion``: columns x, y, z, w of the body-to-navigation rotation. ``euler``: roll,
        pitch, yaw (ZYX) of the body in the navigation frame; ``degrees`` selects the unit.
        ``navigation_frame`` is ``NED`` or ``ENU`` (local, true north).
    accuracy
        Optional standard deviations: ``position_std_columns`` (north, east, down) [m] and
        ``rotation_std_columns`` (roll, pitch, yaw) [rad or deg]; they become pose covariances
        (first order, treating the angles as ZYX Euler angles in the navigation frame).
    """

    timestamp_column: str = "timestamp"
    time_unit: str = "us"
    position: str = "geodetic"
    position_columns: tuple[str, str, str] = ("latitude", "longitude", "altitude")
    crs: str = "EPSG:4979"
    height: str = "ellipsoidal"
    geoid_grid: str | None = None
    undulation_column: str | None = None
    orientation: str = "quaternion"
    orientation_columns: tuple[str, ...] = ("quat.x", "quat.y", "quat.z", "quat.w")
    degrees: bool = False
    navigation_frame: str = "NED"
    position_std_columns: tuple[str, str, str] | None = None
    rotation_std_columns: tuple[str, str, str] | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.position not in ("geodetic", "projected"):
            raise ValueError("position must be 'geodetic' or 'projected'")
        if self.orientation not in ("quaternion", "euler"):
            raise ValueError("orientation must be 'quaternion' or 'euler'")
        expected = 4 if self.orientation == "quaternion" else 3
        if len(self.orientation_columns) != expected:
            raise ValueError(f"{self.orientation} orientation needs {expected} columns")
        if self.navigation_frame not in ("NED", "ENU"):
            raise ValueError("navigation_frame must be 'NED' or 'ENU'")

    @property
    def height_reference(self) -> HeightReference:
        if self.height == "geoid":
            if not self.geoid_grid:
                raise ValueError("height 'geoid' needs geoid_grid")
            return HeightReference.geoid(self.geoid_grid)
        return HeightReference(self.height)


def _read_columns(path: Path) -> dict[str, list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = [name.strip() for name in next(reader)]
        rows = list(reader)
    if not rows:
        raise ValueError(f"'{path}' contains no rows")
    return {name: [row[i] for row in rows] for i, name in enumerate(header)}


def read_pose_csv(path: str | Path, spec: PoseCsvSpec, world: WorldFrame) -> PoseTable:
    """Read a pose CSV into a :class:`~scar.sources.poses.PoseTable` in the world frame (float64)."""
    path = Path(path)
    columns = _read_columns(path)

    def column(name: str) -> np.ndarray:
        if name not in columns:
            raise KeyError(f"Column '{name}' not in {path} (available: {sorted(columns)})")
        return np.asarray(columns[name], dtype=np.float64)

    timestamps = to_nanoseconds(np.asarray(columns[spec.timestamp_column]), spec.time_unit)
    order = np.argsort(timestamps, kind="stable")
    a, b, h = (column(name)[order] for name in spec.position_columns)
    timestamps = timestamps[order]
    undulation = column(spec.undulation_column)[order] if spec.undulation_column else None

    if spec.orientation == "quaternion":
        R_nav_body = quat_xyzw_to_matrix(np.column_stack([column(name)[order] for name in spec.orientation_columns]))
        euler = None
    else:
        euler = [column(name)[order] for name in spec.orientation_columns]
        if spec.degrees:
            euler = [np.radians(angle) for angle in euler]
        R_nav_body = euler_zyx_to_matrix(*euler)

    if spec.position == "geodetic":
        latitude, longitude = a, b
        positions = geodetic_to_world(latitude, longitude, h, world=world, height_reference=spec.height_reference,
            source_crs=spec.crs, undulation=undulation)
    else:
        from pyproj import CRS, Transformer

        source = CRS.from_user_input(spec.crs)
        if not source.is_projected:
            raise ValueError(f"position 'projected' needs a projected CRS, got {spec.crs}")
        # geographic coordinates are needed for height conversion and meridian convergence
        longitude, latitude = Transformer.from_crs(source, source.geodetic_crs, always_xy=True).transform(a, b)
        if source.equals(world.pyproj_crs):
            x, y = a, b
        else:
            x, y = Transformer.from_crs(source, world.pyproj_crs, always_xy=True, allow_ballpark=False,
                only_best=True).transform(a, b)
        heights = convert_height(latitude, longitude, h, spec.height_reference, world, undulation)
        positions = world.to_local(np.column_stack([x, y, heights]))

    R_world_nav = world_from_navigation_rotation(latitude, longitude, world, spec.navigation_frame)
    T_world_body = make_transform(R_world_nav @ R_nav_body, positions)

    covariance = None
    if spec.position_std_columns and spec.rotation_std_columns:
        if euler is None:
            from scipy.spatial.transform import Rotation

            yaw, pitch, roll = Rotation.from_matrix(R_nav_body).as_euler("ZYX").T
            euler = [roll, pitch, yaw]
        position_std = [column(name)[order] for name in spec.position_std_columns]
        rotation_std = [column(name)[order] for name in spec.rotation_std_columns]
        if spec.degrees:
            rotation_std = [np.radians(std) for std in rotation_std]
        stds = np.concatenate([np.ravel(std) for std in position_std + rotation_std])
        if not np.all(np.isfinite(stds) & (stds > 0)):
            # e.g. early internal LTC exports have position_accuracy.* = 0 in every row
            raise ValueError("accuracy columns contain zero, negative or missing standard deviations; "
                "remove position_std_columns/rotation_std_columns to use the preset's prior sigmas")
        covariance_nav = ins_accuracy_covariance(*euler, *rotation_std, *position_std)
        if spec.navigation_frame == "ENU":
            # accuracy columns are given as north, east, down; express them in ENU axes
            swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
            covariance_nav[:, 3:, 3:] = swap @ covariance_nav[:, 3:, 3:] @ swap.T
        A = np.zeros((len(timestamps), 6, 6))
        A[:, :3, :3] = R_world_nav
        A[:, 3:, 3:] = R_world_nav
        covariance = A @ covariance_nav @ np.swapaxes(A, 1, 2)

    unique = np.concatenate([[True], np.diff(timestamps) > 0])
    return PoseTable(timestamps_ns=timestamps[unique], T_world_body=T_world_body[unique],
        covariance=None if covariance is None else covariance[unique])
